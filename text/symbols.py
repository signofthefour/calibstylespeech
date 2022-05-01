# -*- coding: utf-8 -*-
'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

_pad        = '_'
_eos        = '~'
#_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '
_characters   = ' phone1 phone2 phone3 phone4' \

# PLEASE NOTE THAT THIS _CHARACTERS SET WILL NOT BE PROVIDED DUE TO LIMITATION OF INDUSTRY
# READER CAN REIMPLEMENT THIS USE YOUR OWN CHARACTORS SET, BE CAREFUL, YOU MUST READ ALL THE CODE
# BEFORE START ANY WORK WITH THIS,

# EACH PHONEME SET INCLUDE THE ORDER OF EACH PHONE ON IT AFFECT THE FINAL FUNCTION (PRETRAINED MODEL), IT'S JUST A MAPPING
# SO, YOU MUST ENSURE ANY STEP THAT YOU DO


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = _characters.split() 
#print(len(symbols))
