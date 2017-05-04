import json
import random
import data_profiling
'''
json1_file = open('data.json').read()
json2_file = open('data1.json').read()

json1_data = json.loads (json1_file)
json2_data = json.loads(json2_file)



for k,v in json1_data.iteritems():
    print k
print json1_data
'''


data1_file = open('datafile.json').read()
data2_file = open('datafile1.json').read()

json_data1 = json.loads(data1_file)
json_data2 = json.loads(data2_file)

merged_list = json_data1 + json_data2
random.shuffle(merged_list)

list1 = merged_list[0:5000]
list2 = merged_list[5001:]

with open('data_stats_rand1.json', 'w') as outfile:
    json.dump(list1,outfile)
with open('data_stats_rand2.json', 'w') as outfile:
    json.dump(list2,outfile)

data_profiling.driver('data_stats_rand1.json','data_stats1.json')
data_profiling.driver('data_stats_rand2.json','data_stats2.json')