import warnings; warnings.filterwarnings('ignore')
from recommend import RouteRecommender
r = RouteRecommender()
print("Bangalore rate:", r.city_cab_stats.get('Bangalore'))
print("Delhi rate:    ", r.city_cab_stats.get('Delhi'))
print()
for pref in ['Cheapest','Fastest','Economical']:
    res = r.get_best_route('Delhi','Bangalore','Rohini','Whitefield','2026-06-12', pref)
    mode = next((l['Transport_Type'] for l in res['Route Details'] if l['Transport_Type'] != 'Cab'), 'Cab')
    cost = res['Total Cost']
    time = res['Estimated Time']
    print(f"{pref:<22} {mode:<10} {cost:>10}  {time}")
