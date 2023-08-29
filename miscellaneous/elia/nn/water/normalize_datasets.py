from miscellaneous.elia.nn import compute_normalization_factors, normalize

def normalize_datasets(datasets):
    train_dataset = datasets["train"]
    val_dataset   = datasets["val"]
    test_dataset  = datasets["test"]

    print("computing normalization factors for the 'dipole' variable of the train dataset")
    mu, sigma     = compute_normalization_factors(train_dataset,"dipole")
    print("dipole mean :",mu)
    print("dipole sigma:",sigma)

    print("nomalizing the 'dipole' variable of all the dataset")
    train_dataset = normalize(train_dataset,mu,sigma,"dipole")
    val_dataset   = normalize(val_dataset,  mu,sigma,"dipole")
    test_dataset  = normalize(test_dataset ,mu,sigma,"dipole")

    print("final mean and std of the 'dipole' variable of all the dataset")
    mu, sigma     = compute_normalization_factors(train_dataset,"dipole")
    print("train :",mu,",",sigma)
    mu, sigma     = compute_normalization_factors(val_dataset  ,"dipole")
    print("val   :",mu,",",sigma)
    mu, sigma     = compute_normalization_factors(test_dataset ,"dipole")
    print("test  :",mu,",",sigma)

    datasets = {"train":train_dataset,\
            "val"  :val_dataset,\
            "test" :test_dataset }

    return mu, sigma, datasets