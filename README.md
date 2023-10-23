# Script to test clustering algorithms to placement of Data Aggregation Points (DAPs)

1. Add code to scratch folder of the NS-3
2. Set variables in the dap_vars file
3. Open the terminal and access the project root folder 

  - Step 1:
```bash
chmod +x dap_config.py
```
  - Step 2:
```bash
./dap_config.py
```

  - Step 3 (Optional): If any error to be presented, so check the first instruction on the script and swap to python instead python3
```bash
#!/usr/bin/python
```
4. Make test script and run this

## ToDos

1. [X] ~~Add elbow method based on the silhouette score (silhouette score almost always computes k equals to 10 or 15)~~.
2. [X] ~~Add elbow method based on the calinski harabasz score (calinski score almost always computes k equals to 10 or 15)~~.
3. [X] ~~Add shapiro-wilk and lilliefors normal tests.~~
4. [X] ~~Add WCSS analysis for the clusters.~~
5. [ ] Add parameter for simulation script set up the connections between the GWs and their respectives EDs.
6. [ ] Create method in the lorawan-mac-helper file to set up the ED SF based on the conections established.
7. [ ] Create K-Means applying RSSI values as inputs.
8. [ ] Create K-Medoids applying RSSI values as inputs.
9. [ ] Create Fuzzy C-Means applying RSSI values as inputs.
10. [ ] Create Gustafson-Kessel applying RSSI values as inputs.
11. [ ] Create clustering algorithm based on the NSGA-II, applied to define the GWs placements, in order to maximize RSSI and UL-PDR metrics.
12. [ ] Create clustering algorithm based on the MO-PSO, applied to define the GWs placements, in order to maximize RSSI and UL-PDR metrics.