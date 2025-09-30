def lex_leq(tuple1, tuple2):
    assert len(tuple1) == len(tuple2) , "lex_leq, les tuples doivent avoir la meme taille"
    for i in range(len(tuple1)):
        if tuple1[i] <= tuple2[i]:
            return True
        elif tuple1[i] > tuple2[i]:
            return False