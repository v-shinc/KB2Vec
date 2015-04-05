__author__ = 'chensn456'

from kb2vec import KB2Vec,KBTriple,Vocab
import argparse
def test(model_name):
    model = KB2Vec.load(model_name)
    while True:
        entity = raw_input("input entity: ")
        if entity == "exit":
            break
        if entity not in model.vocab:
            print "%s doesn't exist" % entity
            break
        print model.most_similar_entity([entity])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",help = "model name")
    args = parser.parse_args()
    test(args.model)