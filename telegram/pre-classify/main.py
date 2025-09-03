from utils import main


if __name__ == "__main__":
    main(model ="tum-nlp/bertweet-sexism",
         task="sexism")
    
    main(model ="Hate-speech-CNERG/bert-base-uncased-hatexplain",
         task="hatespeech")

    main(model ="mediabiasgroup/magpie-babe-ft",
         task="lexbias")

    main(model ="cardiffnlp/twitter-roberta-base-sentiment-latest",
         task="sentiment")
    

