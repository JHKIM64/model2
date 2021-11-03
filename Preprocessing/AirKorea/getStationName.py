import requests
from bs4 import BeautifulSoup
import pandas as pd

Encoding = 'TTb2xvR9D%2BEc9ydhumshJAa50AiEKEXPJNLuAUuMnU2P0oKHfE8xaZKcojj0Frp2IUg1Tkn1FTpHTB%2FiQKbxbA%3D%3D'
Decoding = 'TTb2xvR9D+Ec9ydhumshJAa50AiEKEXPJNLuAUuMnU2P0oKHfE8xaZKcojj0Frp2IUg1Tkn1FTpHTB/iQKbxbA=='

given_url = 'http://apis.data.go.kr/B552584/MsrstnInfoInqireSvc/getMsrstnList?numOfRows=608&serviceKey='+Encoding

def getInfo(url) :
    Stations = []
    latitudes = []
    longitudes = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text,"html.parser")

    nameList = soup.findAll('stationname')
    latList = soup.findAll('dmx')
    lonList = soup.findAll('dmy')

    for i in range(0,len(nameList)) :
        Stations.append(nameList.__getitem__(i).get_text())
        latitudes.append(latList.__getitem__(i).get_text())
        longitudes.append(lonList.__getitem__(i).get_text())

    StationsInfo=pd.DataFrame(index=Stations,data={"Lon":longitudes,"Lat":latitudes,})
    StationsInfo.index.name='Station'
    # print(StationsInfo)
    StationsInfo.to_csv("/home/intern01/jhk/AirKorea/Korea_stninfo.csv", mode='w')
    return StationsInfo