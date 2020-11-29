import codecs
import glob
import numpy as np
from scipy.io.wavfile import write
import tensorflow.keras.backend as K
from tqdm import tqdm
import sys
from dataloader import *
from hyperparams import Hyperparams as hp
from modules import *
from utils import text_normalize
from IPython.display import Audio

##### Define function for synthesis

def synthesizing(text_to_synthesize,
                 hp,  # hyperparameters
                 decoder_timestep=80):  # Timesteps of decoder

    ##### Get dataloader and a batch of data
    """
    The purpose of getting dataloader is not to use actual data but to use some parameters in it.
    """
    dl = DataLoader(hp)

    for x, y, z in dl.loader:
        break

    ##### Parse

    # Case when using Korean
    if hp.source == "vivos":
        text = text_normalize(text_to_synthesize, hp)
        # text = cleaners.korean_cleaners(text)

    # Case when using English
    else:
        text = text_normalize(text_to_synthesize, hp) + "E"  # E: EOS; end of the sentence            

    texts_synth = [dl.char2idx[char] for char in text]

    ##### Reload model and weights trained
    encoder = get_encoder(hp)

    if hp.use_monotonic and hp.normalize_attention:
        attention_mechanism = BahdanauMonotonicAttention(hp.embed_size, normalize=True)

    elif hp.use_monotonic and not hp.normalize_attention:
        attention_mechanism = BahdanauMonotonicAttention(hp.embed_size)

    elif not hp.use_monotonic and hp.normalize_attention:
        attention_mechanism = BahdanauAttention(hp.embed_size, normalize=True)

    elif not hp.use_monotonic and not hp.normalize_attention:
        attention_mechanism = BahdanauAttention(hp.embed_size)

    decoder1 = AttentionDecoder(attention_mechanism,
                                hp)

    decoder2 = get_decoder2(hp)

    #####
    indexfile = glob.glob(os.path.join(hp.model_dir, "encoder/*.index"))
    indexlist = [int(i_f.split(r"_")[-1].split(r".")[0]) for i_f in indexfile]
    step_index = max(indexlist)

    n_epoch = 1 + (step_index // dl.total_batch_num)

    encoder.load_weights(os.path.join(hp.model_dir, "encoder/weights_{}".format(step_index)))
    decoder1.load_weights(os.path.join(hp.model_dir, "decoder1/weights_{}".format(step_index)))
    decoder2.load_weights(os.path.join(hp.model_dir, "decoder2/weights_{}".format(step_index)))

    print("===== Model upload for synthesis has been completed (step_index = {} // n_epoch = {})".format(step_index,
                                                                                                         n_epoch))

    ##### Generate audio for test sentence

    texts_synth = np.array(texts_synth).reshape(1, -1)

    ##### Set timelength
    Tx = texts_synth.shape[1]
    Ty = decoder_timestep

    ##### Synthesis
    attention_plot_synth = np.zeros((len(texts_synth),
                                     Tx,  # Timestep of text
                                     Ty))  # Timestep of mel

    ##### Compute memory and initla state for decoder
    memory_synth, memory_state_synth, memory_mask_synth = encoder(texts_synth)

    ##### Define decoder input and initial state
    decoder1_input_init_synth = tf.zeros([len(texts_synth), y.shape[-1]])
    decoder1_attn_state_synth = decoder1.attention_cell.get_initial_state(memory_synth)
    decoder1_decoder_state_synth = decoder1.decoder_cell.get_initial_state(memory_synth)
    initial_alignments_synth = decoder1._initial_alignments(len(texts_synth),
                                                            Tx,
                                                            dtype=tf.float32)

    decoder1_state_synth = [decoder1_attn_state_synth,
                            decoder1_decoder_state_synth,
                            initial_alignments_synth,
                            memory_synth,
                            memory_mask_synth]

    decoder1_input_synth = decoder1_input_init_synth
    decoder1_output_synth = []

    for t in tqdm(range(Ty)):  # Iterate based on timestep of the batch (the longest timestep of the batch)

        d_synth, decoder1_state_synth = decoder1(decoder1_input_synth,
                                                 decoder1_state_synth)

        alignments_synth = decoder1_state_synth[2]

        ##### Appending mel-spectogram feature prediction
        decoder1_output_synth.append(d_synth)

        ##### Upldate input and attention plot
        decoder1_input_synth = tf.squeeze(d_synth, axis=1)
        attention_plot_synth[:, :, t] = alignments_synth.numpy()

    mel_synth = tf.concat(decoder1_output_synth, axis=1)

    ##### Calculate loss1 (mel-spectrogram)
    mag_synth = decoder2(mel_synth)

    ##### Saving soundfile and attention plot
    if not os.path.exists("synthesis_{}".format(step_index)):
        os.mkdir("synthesis_{}".format(step_index))

    ind = 0
    for a, m in zip(attention_plot_synth,
                    mag_synth):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(a, cmap='viridis')
        plt.savefig("./synthesis_{}/alignments_{}.png".format(step_index, text_to_synthesize), format="png")

        audio_synth = spectrogram2wav(m.numpy())
        write("./synthesis_{}/audio_synth_{}.wav".format(step_index, text_to_synthesize), hp.sr, audio_synth)

        Audio(url="./synthesis_{}/audio_synth_{}.wav".format(step_index, text_to_synthesize))
        ind += 1


if __name__ == "__main__":
    try:
        hp.source = sys.argv[1]
    except:
        print('SETTING SOURCE DATA FAIL')
    hp.log_dir += hp.source + '/tacotron1_log'
    hp.model_dir += hp.source + '/tacotron1_saved'
    try:
        raw_text = sys.argv[2]
        if isinstance(raw_text, str):
            if os.path.isfile(raw_text):
                with open(raw_text, 'r', encoding='utf-8') as rf:
                    raw_text = rf.read()
    except:
        if hp.source == 'LJSpeech':
            raw_text = 'What are you doing ?\nCan i help you now ?'
        else:
            raw_text = 'Chào mừng đến với hệ thống chuyển văn bản thành giọng nói của tô' \
                       '\n Mọi ý kiến đóng góp liên hệ tôi Nguyễn Mạnh Cường'
        print('SETTING TEST FILE OR RAW TEXT FAILED, REDIRECT TO DEFAULT')
    raw_text = ' . '.join(raw_text.split('\n'))

    # Use different symbols if training with Korean dataset
    hp.use_monotonic = True
    hp.normalize_attention = True

    # Set hp.vocab; originally for LJSpeech (English) dataset. If Korean, symbols need to be changed
    if hp.source == "LJSpeech":
        hp.data += hp.source + '/'
        dl = DataLoader(hp)

    else:
        hp.vocab = "PE abcdeghijklmnopqrstuvxy'.?ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
        hp.data += hp.source + '/train/'
        dl = DataLoader(hp)

    synthesizing(raw_text, hp)
