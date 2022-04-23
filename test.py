import unidecode
from underthesea import word_tokenize

# print (word_tokenize("xin chào"))
for w in word_tokenize("xin chào"):
    print(unidecode.unidecode(w).replace(" ",""))