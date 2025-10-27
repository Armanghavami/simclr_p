





config = {

    'learning_rate' : 0.1,
    'batch_size':256,#256
    'num_epochs':100,
    'image_size':32,

    'size_dataset_train':1, # this is how much of 50000 img is being used to train 

    "lr_factor":0.5,
    "lr_patioence":3,
    "early_stopping_delta":0,
    "early_stopping_patience":10,

    

}



'''



config = {

    'learning_rate' : 0.01,
    'batch_size':50,
    'num_epochs':3,
    'image_size':32,

    'size_dataset_train':0.5, # this is how much of 50000 img is being used to train 

    "lr_factor":0.5,
    "lr_patioence":3,
    "early_stopping_delta":0,
    "early_stopping_patience":20,

    

}
'''