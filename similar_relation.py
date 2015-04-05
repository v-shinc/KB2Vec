__author__ = 'chensn456'

from kb2vec import KB2Vec,Vocab
import  argparse
def test(model_name):
    model = KB2Vec.load(model_name)
    while True:
        relation = raw_input("input relation: ")
        if relation == "exit":
            break
        if relation not in model.vocab_rel:
            print "%s doesn't exist" % relation
            continue
        print model.most_similar_relation(relation,topn=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",help = "model name")
    args = parser.parse_args()
    test(args.model)