from miscellaneous.elia.nn import compute_normalization_factors, normalize

def normalize_datasets(datasets):
    
    print("\n\tNormalizing datasets:")

    train_dataset = datasets["train"]
    val_dataset   = datasets["val"]
    test_dataset  = datasets["test"]

    normalization_factors = {"dipole":None,"energy":None}

    for var in normalization_factors.keys():

        print("\tcomputing normalization factors for the '{:s}' variables of the train dataset".format(var))
        mu, sigma     = compute_normalization_factors(train_dataset,var)
        print("\t\t{:s} mean:".format(var),mu)
        print("\t\t{:s} std :".format(var),sigma)

        if var == "dipole" :
            normalization_factors[var] = {"mean":list(mu),"std":list(sigma)}
        else :
            normalization_factors[var] = {"mean":mu,"std":sigma}

        print("\tNomalizing the '{:s}' variable of all the datasets".format(var))
        train_dataset = normalize(train_dataset,mu,sigma,var)
        val_dataset   = normalize(val_dataset,  mu,sigma,var)
        test_dataset  = normalize(test_dataset ,mu,sigma,var)

        print("\tFinal mean and std of the '{:s}' variable of all the datasets".format(var))
        mu, sigma     = compute_normalization_factors(train_dataset,var)
        print("\t\ttrain :",mu,",",sigma)
        mu, sigma     = compute_normalization_factors(val_dataset  ,var)
        print("\t\tval   :",mu,",",sigma)
        mu, sigma     = compute_normalization_factors(test_dataset ,var)
        print("\t\ttest  :",mu,",",sigma)

        datasets = {"train":train_dataset,\
                    "val"  :val_dataset,\
                    "test" :test_dataset }

    return normalization_factors, datasets