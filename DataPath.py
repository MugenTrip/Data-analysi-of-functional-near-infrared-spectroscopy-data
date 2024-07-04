import os
from queue import LifoQueue

class DataPath:
    """
    Class to finds and store all the paths of the snirf files.
    """
    
    def __init__(self, baseline_path: str, fileType: str="snirf",  recursive: bool=True) -> None:
        self.stack = LifoQueue(maxsize=100)
        self.data_path = list()
        self.baseline_path = baseline_path
        self.stack.put(self.baseline_path)
        self.iter = 0
        if fileType != "snirf" and fileType != "npy":
            print("Only snirf an npy file supported.")
            return -1
        self.fileType = fileType
        if recursive:
            self.recurrentDirSearch()
        else:
            self.getAllinOneDir()
    
    def getAllinOneDir(self):
        '''
        Search for .snirf files in directoty.
        ''' 
        onlyfiles = self.get_immediate_files(self.baseline_path)
        for file in onlyfiles:
            if file.find("." + self.fileType) != -1: 
                self.data_path.append(os.path.join(self.baseline_path,file))

    def get_immediate_subdirectories(self, a_dir):
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    
    def get_immediate_files(self, a_dir):
        return [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]

    def isThisTheFinalDir(self, a_dir):
        onlyfiles = self.get_immediate_files(a_dir)
        for file in onlyfiles:
                if file.find("." + self.fileType) != -1:
                    return os.path.join(a_dir,file)
        return None
    
    def recurrentDirSearch(self):
        '''
        Search recursively all subdirectories for an snirf file. Once it finds a file in a directory it stops looking for another at the same directory.
        ''' 
        self.iter += 1 
        if self.stack.empty():
            return self.data_path
        else:
            a_dir = self.stack.get()
            file = self.isThisTheFinalDir(a_dir)
            if file is not None:
                self.data_path.append(file)
            else:
                subDirs = self.get_immediate_subdirectories(a_dir)
                if subDirs is not None:
                    for dir in subDirs:
                        self.stack.put(dir)
            return self.recurrentDirSearch()
        
    def getDataPaths(self):
        '''
        Return a list wiht all the paths found.
        '''
        return self.data_path