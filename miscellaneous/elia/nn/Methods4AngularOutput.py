import torch
from miscellaneous.elia.good_coding import froze
from abc import ABC, abstractproperty
from typing import TypeVar 
T = TypeVar('T', bound='Methods4AngularOutput')

# @froze
class Methods4AngularOutput(ABC):

    def angular_loss(self:T,func:callable=None,**argv)->callable:

        def loss(x:torch.tensor,y:torch.torch)->torch.tensor:

            x = torch.fmod(x,1.0)
            y = torch.fmod(y,1.0)

            delta = x-y
            phases = torch.min( delta , 1-delta )

            if True :
                for element in phases.flatten():
                    if not (0 <= element < 1):
                        raise ValueError("Not all elements are within [0, 1)")
                    
            # The loss has to depend only on the norm
            # Here we are computing the norm squared just for efficiency
            r = torch.square(phases).sum(axis=1)

            if True :
                sqrt2 = torch.sqrt(0.5)
                for element in r.flatten():
                    if not (0 <= element < sqrt2):
                        raise ValueError("Not all elements are within [0, sqrt(2) )")
                    
            if func is not None :
                r = func(r)
            
            return r
        
        return loss 
