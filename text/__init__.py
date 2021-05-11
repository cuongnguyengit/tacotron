""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import get_symbols


class CustomText:
    def __init__(self, symbol_name):
        self.symbols = get_symbols(symbol_name)
        # Mappings from symbol to numeric ID and vice versa:
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        # Regular expression matching text enclosed in curly braces:
        self._curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

    def text_to_sequence(self, text, cleaner_names):
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = self._curly_re.match(text)

            if not m:
                sequence += self._symbols_to_sequence(self._clean_text(text, cleaner_names))
                break
            sequence += self._symbols_to_sequence(self._clean_text(m.group(1), cleaner_names))
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        return sequence


    def sequence_to_text(self, sequence):
        """Converts a sequence of IDs back to a string"""
        result = ""
        for symbol_id in sequence:
            if symbol_id in self._id_to_symbol:
                s = self._id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == "@":
                    s = "{%s}" % s[1:]
                result += s
        return result.replace("}{", " ")


    def _clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
            text = cleaner(text)
        return text


    def _symbols_to_sequence(self, symbols):
        return [self._symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]


    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(["@" + s for s in text.split()])


    def _should_keep_symbol(self, s):
        return s in self._symbol_to_id and s != "_" and s != "~"
