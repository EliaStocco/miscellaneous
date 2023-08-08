from torch_geometric.loader import DataLoader

def _make_dataloader(dataset,batch_size=1,shuffle=True,drop_last=True):

        if batch_size == -1 :
                batch_size = len(dataset)
            
        dataloader = DataLoader(dataset,\
                                batch_size=batch_size,\
                                drop_last=drop_last,\
                                shuffle=shuffle)
    
        return dataloader