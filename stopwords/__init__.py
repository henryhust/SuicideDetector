import os
dir_path = os.path.dirname(os.path.abspath(__file__))

STOPWORDS = [line.strip()for line in open(os.path.join(dir_path, "stopwords.txt"), encoding="utf8")]