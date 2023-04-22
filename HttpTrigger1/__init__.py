import logging
import numpy as np
import azure.functions as func
import pandas as pd
import json
from io import BytesIO, StringIO
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

def recommandation_SVD_generator(articles, df, model_b, user_id, n_items):
    logging.info('-----------debut traitement 1')
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
    logging.info('---------------------fin traitement 1')
    logging.info('---------------------debut traitement 2')
    # Obtenir une liste de tous les identifiants de article à partir du jeu de données 
    arts_ids = articles_click_au_moins_une_fois

    # Obtenir une liste de tous les identifiants de article qui ont été cliqué par l'utilisateur 
    arts_ids_user = user_articles_clickes(user_id, all_clicks_df_sav)
    # Obtenir une liste de tous les ID de article qui n'ont pas été cliqué par l'utilisateur 
    arts_ids_to_pred = np.setdiff1d(arts_ids, arts_ids_user["article_id"]) 

    # Appliquer une note de 0 à toutes les interactions (uniquement pour correspondre au format de l'ensemble de données Surprise) 
    test_set = [[user_id, art_id, 0] for art_id in arts_ids_to_pred] 
    logging.info('---------------------fin traitement 2')
    logging.info('---------------------debut traitement 3')
    model0 =  pickle.loads(model_b)
    model = model0["algo"]
    logging.info('---------------------fin traitement 3')
    logging.info('---------------------debut traitement 4')
    # Prédire les notes et générer des recommandations 
    predictions = model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    # Classer les n meilleurs films en fonction des prédictions notes 
    index_max = (-pred_ratings).argsort()[:n_items] 
    logging.info('---------------------fin traitement 4')
    logging.info('---------------------debut traitement 5')
    result = []
    for i in index_max: 
        art_id = arts_ids_to_pred[i] 
        result.append({"article_id":str(all_clicks_df[all_clicks_df["article_id"]==art_id]["article_id"].values[0]) , "predictions":str(pred_ratings[i])})
    logging.info('---------------------fin traitement 5')
    return json.dumps(result)
 

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    blob_client = BlobClient.from_blob_url("https://recommendcosinusap2e95b4.blob.core.windows.net/files/model.pkl?sp=r&st=2023-04-16T00:32:25Z&se=2024-04-17T08:32:25Z&spr=https&sv=2021-12-02&sr=b&sig=BScSV6cldDVoBEVLlCY3i%2BFSjWSgkgsUchhZqjhiXcY%3D")
    download_stream = blob_client.download_blob()
    logging.info('------------debut blob pkl')
    model_b = download_stream.readall()
    logging.info('------------fin blob pkl')
    logging.info('------------debut article')
    blob_client = BlobClient.from_blob_url("https://recommendcosinusap2e95b4.blob.core.windows.net/files/articles_metadata.csv?sp=r&st=2023-04-15T22:04:55Z&se=2024-04-16T06:04:55Z&spr=https&sv=2021-12-02&sr=b&sig=YHnyaPownqSu9x3QpREKe%2FldU%2FEXX5TAHSzf9UXr%2B14%3D")
    blob = blob_client.download_blob().content_as_text()
    articles = pd.read_csv(StringIO(blob))
    logging.info('------------fin article')
    logging.info('------------debut df')
    blob_client = BlobClient.from_blob_url("https://recommendcosinusap2e95b4.blob.core.windows.net/files/df_clicks.csv?sp=r&st=2023-04-16T20:11:57Z&se=2024-04-18T04:11:57Z&spr=https&sv=2021-12-02&sr=b&sig=iSxt%2FE126%2B01EMNaKcjCHzoOIn4kcVriQUJHGemK5Kc%3D")
    blob = blob_client.download_blob().content_as_text()
    df = pd.read_csv(StringIO(blob))
    logging.info('------------fin df')
    req_body_bytes = req.get_body()
    req_body = req_body_bytes.decode("utf-8")
    json_body = json.loads(req_body)
    name = None
    name = json_body['user_id']
    name = int(name)
    logging.info('------------debut reco')
    valeur = recommandation_SVD_generator(articles, df, model_b, name,5)
    logging.info('------------fin reco'+str(valeur))
    func.HttpResponse.charset = 'utf-8'
    return func.HttpResponse(
            valeur,
            status_code=200
            )


