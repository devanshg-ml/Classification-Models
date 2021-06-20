## usage of Tree verification  
1,Open the tree_verification.ipynb with google colab  
2,Upload the train.csv  
3,Run all cells,the accuracy will be shown in the end  

## run on testing data
Based on the load model code, the way to get the prediction data based on the testing data will be
```python
predict=pickle_model.predict(test_set)
prediction = pd.DataFrame(predict, columns=['predictions']).to_csv('prediction.csv',index=False)
```
