from multiprocessing.managers import BaseManager
import Network

class MyManager(BaseManager):
    pass

MyManager.register('Network_copy', Network)