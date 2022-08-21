from texts.corpus import Corpus

inputdir = "/Users/jstegink/thesis/corpora/wikimatch/nl"
output =  "/users/jstegink/weg/secties.txt"
corpus = Corpus(directory=inputdir)

names = {}
for document in corpus:
    for section in document:
        name = section.get_title()
        if name in names:
            names[name] += 1
        else:
            names[name] = 1


for name in names.keys():
    if names[name] > 100:
        print(f"{name}\t{names[name]}")