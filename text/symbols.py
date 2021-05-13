""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """
try:
    from phoneme import list_phones
except:
    from text.phoneme import list_phones

def get_symbols(name):
    if name == 'char':
        _pad = '_'
        _punctuation = ' !,.;?'
        _letters = 'aáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệghiíỉìĩịklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvxyýỷỳỹỵ'
        symbols = [_pad] + list(_punctuation) + list(_letters)
    elif name == 'phoneme':
        _punctuation = ' !,.;?'
        symbols = list(_punctuation) + ['@' + i for i in list_phones]
    return symbols