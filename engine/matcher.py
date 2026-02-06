def match(emb, db_full, db_upper):
    s1 = cosine(emb, db_full)
    s2 = cosine(emb, db_upper)
    return max(s1, s2)
