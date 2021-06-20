## CNN useage
1.Open the network.ipynb with google colab  
2.pload the train.csv  
3.Run all cells,two score will be shown in the end, one is the loss (almost 0) the other is accuracy (almost 1)  

## Run on testing  
The load model part already have the code that will generate the predict file for the testing data.
```python
model= load_model('3CNN.model')
pre=model.predict(test_set.reshape(-1,28,28,1))
predict=np.argmax(pre,axis=1)
prediction = pd.DataFrame(predict, columns=['predictions']).to_csv('prediction.csv',index=False)
```
