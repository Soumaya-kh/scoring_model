
import pickle
import pandas as pd
from flask import Flask, request, render_template
import json
app = Flask(__name__)
app.config["DEBUG"] = True
# chargement du modele
pickle_inp = open('model.pickle', "rb")
classifier = pickle.load(pickle_inp)

# importer les donnees test
x_test = pd.read_csv("data_test.csv")
x_train = pd.read_csv("data_train.csv")
target = pd.read_csv("TARGET.csv")
@app.route('/', methods=['GET'])
def bienvenue():  # put application's code here
    return 'Bienvenue à ma premiere API de ML!!!'

@app.route('/predict/<ID>', methods=['GET'])
def make_predict(ID):
    '''Fonction de prédiction utilisée par l\'API flask :
    a partir de l'identifiant et du jeu de données
    renvoie la prédiction à partir du modèle'''

    ID = int(ID)
    X = x_test[x_test['SK_ID_CURR'] == ID]
    X = X.drop(['SK_ID_CURR'], axis=1)
    proba = classifier.predict_proba(X)[:, 1][0]
    proba_json = {'probabilite': str(proba)}
    from flask import jsonify
    return jsonify(proba_json)

if __name__ == '__main__':
    app.run()
