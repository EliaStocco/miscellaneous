from torch_geometric.loader import DataLoader

def _make_dataloader(dataset,batch_size=1):
            
        dataloader = DataLoader(dataset,\
                                batch_size=batch_size,\
                                drop_last=True,\
                                shuffle=True)
    
        return dataloader