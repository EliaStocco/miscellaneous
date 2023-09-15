from miscellaneous.elia.nn import compute_normalization_factors, normalize

def normalize_datasets(datasets):
    
    print("\n\tNormalizing datasets:")

    train_dataset = datasets["train"]
    val_dataset   = datasets["val"]  if "val"  in datasets else None
    test_dataset  = datasets["test"] if "test" in datasets else None

    normalization_factors = {"dipole":None,"energy":None}

    for var in normalization_factors.keys():

        print("\tcomputing normalization factors for the '{:s}' variables of the train dataset".format(var))
        mu, sigma     = compute_normalization_factors(train_dataset,var)
        print("\t\t{:s} mean:".format(var),mu)
        print("\t\t{:s} std :".format(var),sigma)

        if var == "dipole" :
            mu = 0.

        # if var == "dipole" :
        #     normalization_factors[var] = {"mean":list(mu),"std":list(sigma)}
        # else :
        normalization_factors[var] = {"mean":mu,"std":sigma}

        print("\tNomalizing the '{:s}' variable of all the datasets".format(var))
        train_dataset = normalize(train_dataset,mu,sigma,var)

        if val_dataset is not None :
            val_dataset   = normalize(val_dataset,  mu,sigma,var)
            
        if test_dataset is not None :
            test_dataset  = normalize(test_dataset ,mu,sigma,var)

        print("\tFinal mean and std of the '{:s}' variable of all the datasets".format(var))
        mu, sigma     = compute_normalization_factors(train_dataset,var)
        print("\t\ttrain :",mu,",",sigma)

        if val_dataset is not None :
            mu, sigma = compute_normalization_factors(val_dataset  ,var)
            print("\t\tval   :",mu,",",sigma)

        if test_dataset is not None :
            mu, sigma = compute_normalization_factors(test_dataset ,var)
            print("\t\ttest  :",mu,",",sigma)

        print("\n")

    datasets = {"train":train_dataset}
    
    if val_dataset is not None :
        datasets["val"] = val_dataset

    if val_dataset is not None :
        datasets["test"] = test_dataset       
        

    return normalization_factors, datasets