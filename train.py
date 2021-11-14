
def train(model, ds_train, ds_val):

    model.compile(optimizer='adam', 
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

    model.fit(ds_train, 
                epochs=2, 
                verbose=2,
                validation_data=ds_val,
                validation_steps=1)
