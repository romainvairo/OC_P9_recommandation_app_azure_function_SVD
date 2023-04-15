import logging
import numpy as np
import azure.functions as func
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
from io import BytesIO, StringIO
# predicitng
from pprint import pprint as pp
import pickle
import surprise
from azure.storage.blob import BlobClient


def transform_to_dataframecsv(blob):
    dfs = bytearray(blob.read())
    dfs = pd.read_csv(BytesIO(dfs))
    return dfs

def transform_to_dataframepick(blob):
    dfs = bytearray(blob.read())
    dfs = pd.read_pickle(BytesIO(dfs))
    return dfs

def outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    Q1min = Q1 - (1.5 * IQR)
    Q3max = Q3 + (1.5 * IQR)
    return Q1min, Q3max

def user_articles_clickes(user_id, data):
    article_du_user = data.loc[data["user_id"] == user_id]
    return article_du_user

def recommandation_SVD_generator(articles, df, model, user_id, n_items):
    minScore, maxScore = outlier(articles["words_count"])

    articles["words_count"] = np.where(articles["words_count"] < minScore, minScore, articles["words_count"])
    articles["words_count"] = np.where(articles["words_count"] > maxScore, maxScore, articles["words_count"])

    all_clicks_df = df.join(articles, how='left', on='click_article_id', lsuffix="_1")

    #Create a map to convert article_id to category
    dict_article_categories = articles.set_index('article_id')['category_id'].to_dict()

    #Get Categorie associate for each article
    all_clicks_df['category_id'] = all_clicks_df['click_article_id'].map(dict_article_categories).astype(int)
    all_clicks_df['total_click'] = all_clicks_df.groupby(['user_id'])['click_article_id'].transform('count')
    all_clicks_df['total_click_by_category_id'] = all_clicks_df.groupby(['user_id','category_id'])['click_article_id'].transform('count')
    all_clicks_df['rating'] = all_clicks_df['total_click_by_category_id'] / all_clicks_df['total_click']

    all_clicks_df = all_clicks_df.drop_duplicates()

    all_clicks_df = all_clicks_df.drop(['total_click', 'total_click_by_category_id'], axis=1)

    all_clicks_df_sav = all_clicks_df.copy()

    all_clicks_df = all_clicks_df.drop(['click_article_id'], axis=1)

    articles_click_au_moins_une_fois = all_clicks_df_sav.click_article_id.value_counts().index

    # Obtenir une liste de tous les identifiants de article à partir du jeu de données 
    arts_ids = articles_click_au_moins_une_fois

    # Obtenir une liste de tous les identifiants de article qui ont été cliqué par l'utilisateur 
    arts_ids_user = user_articles_clickes(user_id, all_clicks_df_sav)
    # Obtenir une liste de tous les ID de article qui n'ont pas été cliqué par l'utilisateur 
    arts_ids_to_pred = np.setdiff1d(arts_ids, arts_ids_user["article_id"]) 

    # Appliquer une note de 0 à toutes les interactions (uniquement pour correspondre au format de l'ensemble de données Surprise) 
    test_set = [[user_id, art_id, 0] for art_id in arts_ids_to_pred] 
    model =  pickle.loads(model)
    model = model["algo"]
    # Prédire les notes et générer des recommandations 
    predictions = model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    print("Top {0} recommandations d'articles pour l'utilisateur {1} :".format(n_items, user_id))
    # Classer les n meilleurs films en fonction des prédictions notes 
    index_max = (-pred_ratings).argsort()[:n_items] 
    result = []
    for i in index_max: 
        art_id = arts_ids_to_pred[i] 
        print(all_clicks_df[all_clicks_df["article_id"]==art_id]["article_id"].values[0] , pred_ratings[i])
        result.append({"article_id":all_clicks_df[all_clicks_df["article_id"]==art_id]["article_id"].values[0] , "predictions":str(pred_ratings[i])})
    return result
 

def main(req: func.HttpRequest, dfartblob: func.InputStream, dfblob: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    blob_client = BlobClient.from_blob_url("https://recommendcosinusap2e95b4.blob.core.windows.net/files/model.pkl?sp=r&st=2023-04-15T20:25:51Z&se=2024-04-16T04:25:51Z&sv=2021-12-02&sr=b&sig=uZ2HDbXvmiPeMA18WfvLBwJESxh%2FjdPybNMUGJGF%2BKY%3D")
    download_stream = blob_client.download_blob()
    logging.info('------------blob pkl')
    model = download_stream.readall()
    articles = transform_to_dataframecsv(dfartblob)  # df_clicks.csv
    df = transform_to_dataframecsv(dfblob)  # df_clicks.csv
    req_body_bytes = req.get_body()
    req_body = req_body_bytes.decode("utf-8")
    json_body = json.loads(req_body)
    name = None
    name = json_body['user_id']
    name = int(name)

    return func.HttpResponse(
            json.dumps(recommandation_SVD_generator(articles, df, model, name)),
            status_code=200
            )


