# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:58:13 2019

@author: chigr
"""

import psycopg2
from psycopg2 import Error

connection = psycopg2.connect(user = "postgres", 
                              password = "0888", 
                              host = "127.0.0.1", 
                              port = "5432",
                              database = "HackerRank Challenge")

cursor = connection.cursor()

select_all_challenges = '''
SELECT * FROM CHALLENGES;
 '''
    
cursor.execute(select_all_challenges)
cursor.execute('SELECT * FROM COLLEGES')

connection.commit()

r = cursor.fetchall()

cursor.execute('insert into colleges (college_id) values (123)')
cursor.execute('insert into colleges values (1234, 4566)')

cursor.execute('delete from colleges where college_id in (123,1234)')


connection.commit()

if(connection):
      cursor.close()
      connection.close()

'''
except (Exception, psycopg2.DatabaseError) as error :
    print ("Error while creating PostgreSQL table", error)
finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
'''         


def roadsAndLibraries(n, c_lib, c_road, cities):
    total = 0
    if c_lib < c_road:
        total = n*c_lib
    else:       
        neighbours = {}
        visited = [False] * n
        connectedComponents = 0
        nodes_per_cluster = {}

        #recursive DFS
        def dfs(i,cluster):
            print(f'visited {visited}');
            if not visited[i]:
                #check how many unique nodes are in this cluster
                print(f'nodes_per_cluster {nodes_per_cluster}');
                nodes_per_cluster[cluster] = (
                    nodes_per_cluster.get(cluster,0) + 1)
            #mark this as visited
            visited[i] = True
            my_neighbours = []
            print(f'my_neighbours before {my_neighbours}');
            try:
                my_neighbours = neighbours[i+1] 
                print(f'neighbours  {neighbours}');
            except KeyError as ke:
                # we found a single node cluster (city with one house)
                # leave the list empty and the for-loop will skip it
                pass
            print(f'my_neighbours after {my_neighbours}');
            for city_id in my_neighbours:
                if not visited[city_id-1]:
                    print(f'--> recursive dfs {city_id-1}, {cluster}');
                    dfs(city_id-1,cluster)

        #populate the adjacency list
        for road in cities:
            neighbours[road[0]] = (
                neighbours.get(road[0],[]) + [road[1]])
            neighbours[road[1]] = (
                neighbours.get(road[1],[]) + [road[0]])

        
        for i in range(n):
            if not visited[i]:
                print(f'-> into dfs from main{i}, {i}');
                dfs(i,i)
                connectedComponents += 1

        #min number of roads is always number of houses - 1   
        print(f'nodes_per_cluster.values {nodes_per_cluster.values}')
        for x in nodes_per_cluster.values(): print(x)
        roads = sum(x-1 for x in nodes_per_cluster.values())
            
        total = c_road * roads + c_lib * connectedComponents
    return total



[n, m, c_lib, c_road] = map(int,'5 9 92 23'.split())
cities = []
cities.append(list(map(int, '2 1'.rstrip().split())))
cities.append(list(map(int, '5 3'.rstrip().split())))
cities.append(list(map(int, '5 1'.rstrip().split())))
cities.append(list(map(int, '3 4'.rstrip().split())))
cities.append(list(map(int, '3 1'.rstrip().split())))
cities.append(list(map(int, '5 4'.rstrip().split())))
cities.append(list(map(int, '4 1'.rstrip().split())))
cities.append(list(map(int, '5 2'.rstrip().split())))
cities.append(list(map(int, '4 2'.rstrip().split())))
