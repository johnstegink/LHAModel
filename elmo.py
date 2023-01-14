from elmoformanylangs import Embedder

e = Embedder('/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/ELMoForManyLangs-master/nl')

sents = [['Ik', 'kom', 'uit', 'Bathmen', 'Nederland'],
['Vandaag', 'ben', 'ik', 'in', 'het', 'ziekehuis', 'geweest']]
# the list of lists which store the sentences
# after segment if necessary.

vectors = e.sents2elmo(sents)
# will return a list of numpy arrays
# each with the shape=(seq_len, embedding_size)

a = 0