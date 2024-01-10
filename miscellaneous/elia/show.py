
def show_dict(obj:dict,string:str=""):
    for k in obj.keys():
        print("{:s}{:30s}:".format(string,k),obj[k])