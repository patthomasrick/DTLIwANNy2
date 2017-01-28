import xml2trainingdata as x2t

c = x2t.XMLtoTrainingData()

d = c.load_xml()

for k in d.keys():
    for leaf in d[k]:
        for k2 in leaf:
            print(k2, '\t\t', leaf[k2])