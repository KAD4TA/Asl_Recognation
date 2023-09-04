import os

PATH="./my_DATA"
ALPHABET=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
PARTS=['train','test','validation']

if not os.path.exists(PATH):
    os.makedirs("./DATA")
    for content in PARTS:
        for alp in ALPHABET:
            file_path= os.path.join(PATH,content,alp)
            os.makedirs(file_path)
        
