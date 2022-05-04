with open("彩虹后宫_out_co.txt", "r", encoding="utf8") as fr:
    content = fr.read()
    from smoothnlp.algorithm.phrase import extract_phrase
    new_phrases = extract_phrase(content)
    print(new_phrases)