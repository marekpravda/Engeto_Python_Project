# najprv som si načítal potrebné knižnice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy
from statistics import median

# stiahol som si data z databázy
user = 'student'
password = 'p7@vw7MCatmnKjy7'
conn_string = f"mysql+pymysql://{user}:{password}@data.engeto.com/data"
alchemy_conn = sqlalchemy.create_engine(conn_string)

edinburgh_bikes = pd.read_sql('edinburgh_bikes', alchemy_conn)
# pre zjednodušenie som odstránil momentálne nepotrebné stĺpce z tabuľky
edinburgh_bikes = edinburgh_bikes.drop(columns=['start_station_description', 'end_station_description', 'index'])

# vytvoril som si dataframe s popisnou štatistikou tabuľky, z ktorej som ponechal len štatistiky o dĺžke ciest
# z nej je možné vidieť ako dlho v priemere ľudia cestujú, prípadne koľko trvala najdlhšia, alebo najkratšia cesta
desc_statistics = edinburgh_bikes.describe().round(2)
desc_statistics = desc_statistics['duration']

# v tomto prípade pri tak výraznej odchylke je možné použiť medián, ktorý nie je tak výrazne ovplyvnený extrémnymi hodnotami
median_duration = median(edinburgh_bikes['duration'])
print("Medián dĺžky cesty je ", int(median_duration)//60, "minút")

# ďalej chceme určiť tzv. aktívne a neaktívne stanice, na to je možné využiť počet využití danej stanice za sledované obdobie
start_station_counts = edinburgh_bikes['start_station_id'].value_counts()
start_station_counts = pd.DataFrame(start_station_counts)
start_station_counts.columns = ['Count_started']
start_station_counts.index.name = 'Station_id'
start_station_counts = start_station_counts.rename(columns={"start_station_id" : "Count"})

end_station_counts = edinburgh_bikes['end_station_id'].value_counts()
end_station_counts = pd.DataFrame(end_station_counts)
end_station_counts.columns = ['Count_ended']
end_station_counts.index.name = 'Station_id'
#vyfiltroval som si zvlast aktivne počiatočne stanice
active_start_stations = start_station_counts.loc[start_station_counts['Count_started'] > 776]
active_start = list(active_start_stations.index)
# zvlast konečne
active_end_stations = end_station_counts.loc[end_station_counts['Count_ended'] > 776]
active_end = list(active_end_stations.index)
# nasledne som nasiel prienik - stanice, ktore boli pouzite priemerne aspon raz denne ani na odchod ani na prichod
active_stations = set(active_start + active_end)

# vyfiltroval som si zvlast neaktivne počiatočne stanice
inactive_start_stations = start_station_counts.loc[start_station_counts['Count_started'] < 776]
inactive_start = list(inactive_start_stations.index)
# zvlast konečne neaktivne stanice
inactive_end_stations = end_station_counts.loc[end_station_counts['Count_ended'] < 776]
inactive_end = list(inactive_end_stations.index)
# nasledne som nasiel prienik - stanice, ktore neboli pouzite priemerne aspon raz denne ani na odchod ani na prichod
inactive_stations = set(inactive_start).intersection(inactive_end)

# najfrekventovanejsie stanice
most_frequent_stations = pd.merge(start_station_counts,  end_station_counts, left_index = True, right_index = True)
most_frequent_stations ['Total_count'] = most_frequent_stations['Count_started'] + most_frequent_stations ['Count_ended']
most_frequent_stations = most_frequent_stations.sort_values(by=['Total_count'], ascending= False)
most_frequent_stations.index.name = 'Station_id'
most_frequent_stations = most_frequent_stations.head(10)
most_frequent_stations ['Average'] = most_frequent_stations['Total_count'] / 776
most_frequent_stations ['Average'] = most_frequent_stations ['Average'].round().astype('int64')
print(most_frequent_stations)
# z tabulky je mozne vidiet celkovy pocet ciest pre ktore boli dane stanice počiatočné, alebo konečné
# okrem toho je vidieť súčasťou koľkých ciest bola daná stanica priemerne za deň
# vybral som top 10 staníc z pohľadu množstva ciest ktoré danú stanicu obsahovali

# # hromadenie bicyklov
hoarding_bicycles = pd.merge(start_station_counts,  end_station_counts, left_index = True, right_index = True)
hoarding_bicycles ['Difference'] = hoarding_bicycles['Count_ended'] - hoarding_bicycles ['Count_started']
hoarding_bicycles = hoarding_bicycles.sort_values(by=['Difference'])
hoarding_bicycles['Average'] = hoarding_bicycles['Difference'] / 776
hoarding_bicycles ['Average'] = hoarding_bicycles['Average'].round().astype('int64')

#  podmienky pre hromadenie som si zvolil tak aby na danej stanici zostával navyše aspoň jeden bicykel
hoarding = hoarding_bicycles[hoarding_bicycles['Difference'] > 776]
hoarding = hoarding.sort_values(by=['Difference'], ascending= False)
#  v troch pripadoch bol počet prichádzajúcich bicyklov rovnaký ako odchádzajúcich a preto som ich zahrnul do zvlášť kategorie
neutral = hoarding_bicycles[hoarding_bicycles['Difference'] == 0]
#  pre stanice, ktore by som označil ako tie na ktorych chybaju bicykle som urcil podmienky tak aby na danej stanici priemerne ubudlo o jeden bicykel viac nez pribudlo
missing = hoarding_bicycles[hoarding_bicycles['Difference'] < -776]

# vypocet vzdialenosti medzi stanicami
def haversine_np(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

edinburgh_bikes['Distance'] = haversine_np(edinburgh_bikes['start_station_longitude'],edinburgh_bikes['start_station_latitude'],edinburgh_bikes['end_station_longitude'],edinburgh_bikes['end_station_latitude'])
edinburgh_bikes = edinburgh_bikes.sort_values(by='Distance',ascending=False)

# 3 ODLAHLE HODNOTY SU Z LIVERPOOLU
# odstranil som ich kvoli skresleniu
edinburgh_bikes.drop(edinburgh_bikes[edinburgh_bikes.end_station_id == 280].index, inplace=True)
# nulove hodnoty znamenaju ze niekto zacal aj skoncil na tom istom mieste

edinburgh_bikes['duration_in_min'] = edinburgh_bikes['duration'] / 60
edinburgh_bikes['duration_in_min'] = edinburgh_bikes['duration_in_min'].astype("int")
# zoradil som si hodnoty podla dlzky trvania cesty a odstranil som tie ktore trvali dlhsie nez 120 minut, nakolko sa vyssie
# hodnoty vyskytovali len velmi malokedy a vykreslenie histogramu bolo preto komplikovane
edinburgh_bikes = edinburgh_bikes.sort_values(by=['duration'], ascending= False)
edinburgh_bikes = edinburgh_bikes[edinburgh_bikes['duration_in_min'] < 120 ]

# nasledne som si vykreslil histogram s upravenymi datami

plt.rcParams["figure.figsize"] = (7,5)
edinburgh_bikes.hist(column='duration_in_min', bins= 10)
plt.xlim(0,120)
plt.title('Histogram of the trip durations')
plt.xlabel('Duration in minutes')
plt.ylabel('Number of trips')
plt.show()

edinburgh_bikes[['start_date','start_time']] = edinburgh_bikes['started_at'].astype(str).str.split(' ',expand=True)
demand = edinburgh_bikes.groupby('start_date').count()
print(demand['started_at'].equals(demand['ended_at']))
# overil som si, ze su dane stlpce totozne a ponechal som si len jeden
demand = demand [['started_at']]
demand.columns = ['Demand']
demand.index = pd.to_datetime(demand.index, dayfirst=True)
demand_gb = demand.groupby([demand.index.year, demand.index.month]).sum()

# na oznacenie osi som potreboval zmenit format datumu na mesiac/rok
index_list = []
for index in demand_gb.index:
    index_list.append (list(index))

date_list = []
for y,m in index_list:
    date_list.append(f"{m}/{y}")

demand_gb.index = demand_gb.index.droplevel()
demand_gb.index = date_list
# GRAF poptavka za mesiac
plt.rcParams["figure.figsize"] = (10,9)
demand_gb.plot(legend= None)
plt.title('Demand by months')
plt.xticks(np.arange(0,26),date_list,rotation=90)
plt.xlabel(" ")
plt.ylabel("Number of bike rides")
plt.grid()
plt.savefig("C:\\Users\\Marek\\Desktop\\Engeto\\Data_academy\\Projekt_python\\demand.jpg")
plt.show()

# nacitanie dat o pocasi

edinburgh_weather = pd.read_sql('edinburgh_weather', alchemy_conn)

# vypocet priemernej mesacnej dennej teploty
# ako ukazovatel teploty som pouzil udaj o pocitovej teplote, ktora zohladnuje okrem teploty aj silu vetra a vlkost vzduchu

edinburgh_weather = edinburgh_weather[edinburgh_weather['time'].isin(['06:00', '09:00','12:00','15:00','18:00'])]
edinburgh_weather.index = edinburgh_weather ['date']
edinburgh_weather.index = pd.to_datetime(edinburgh_weather.index,format= "%Y-%m-%d")
edinburgh_weather['feels'] = edinburgh_weather['feels'].str.strip('°c')
edinburgh_weather['feels'] = edinburgh_weather['feels'].astype(int)
edinburgh_weather.rename(columns={"feels": "Feel"},inplace = True)
temp_gb = edinburgh_weather.groupby([edinburgh_weather.index.year, edinburgh_weather.index.month])['Feel'].mean()
temp_gb = pd.DataFrame(temp_gb)
temp_gb.index = temp_gb.index.droplevel()
temp_gb.index = date_list

demand_weather = pd.merge(demand_gb,  temp_gb,left_index = True, right_index= True)
# vykreslenie grafov dopytu a teploty
fig, ax1 = plt.subplots()
fig.suptitle("Influence of weather on trip demand", fontsize=16)
color = 'tab:red'
ax1.set_xlabel('Month', labelpad= 25)
ax1.set_ylabel('Demand', color="black", labelpad= 25 )
ax1.plot(demand_weather.index, demand_weather['Demand'], color=color)
ax1.tick_params(axis='x', labelcolor="black", labelrotation=90)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average month temperature', color="black",labelpad= 25)
ax2.plot(demand_weather.index, demand_weather['Feel'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

correlation = demand_weather['Demand'].corr(demand_weather['Feel'])
print("Korelacia priemernej pocitovej teploty a dopytu je:", round(correlation, 2))

# vikend vs pracovny tyzden
edinburgh_bikes.index = edinburgh_bikes['started_at']
edinburgh_bikes.index = pd.to_datetime(edinburgh_bikes.index)
edinburgh_bikes['Weekend'] = ((pd.DatetimeIndex(edinburgh_bikes.index).dayofweek) // 5 == 1).astype(int)
weekend_gb = edinburgh_bikes.groupby(edinburgh_bikes['Weekend'])['started_at'].count()
weekend_gb = pd.DataFrame(weekend_gb)
weekend_gb.columns = ['Demand']

# najprv som si spocital pocet tyzdnov aby som dokazal spocitat pocet vikendov a pocet pracovnych dni
edinburgh_bikes['started_at'] = edinburgh_bikes['started_at'].astype("datetime64")
days = (max(edinburgh_bikes['started_at']) - min(edinburgh_bikes['started_at'])).days
weeks = days//7
weekends = weeks * 2
work_days = weeks * 5

# spocital som priemerny pocet ciest cez vikend a cez pracovny tyzden
weekend_gb['Average_demand'] = weekend_gb['Demand']
weekend_gb.iloc[0, weekend_gb.columns.get_loc('Average_demand')] = (weekend_gb.iloc[0, weekend_gb.columns.get_loc('Demand')])/work_days
weekend_gb.iloc[1, weekend_gb.columns.get_loc('Average_demand')] = (weekend_gb.iloc[1, weekend_gb.columns.get_loc('Demand')])/weekends
w_list = ['Weekdays','Weekends']
weekend_gb.index = w_list
#vysledkom, je ze viac ludi vyuzivalo bicykel cez vikend nez pocas pracovneho tyzdna

plt.rcParams["figure.figsize"] = (7,7)
weekend_gb['Average_demand'].plot.bar(color=['red', 'blue'])
plt.title('Average demand')
plt.xticks(rotation=0)
plt.show()
