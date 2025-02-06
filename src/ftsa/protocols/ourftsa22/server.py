from collections import defaultdict
from math import ceil, factorial
from gmpy2 import mpz

from ftsa.protocols.buildingblocks.utils import subs_vectors, powmod
from ftsa.protocols.buildingblocks.PRG import PRG
from ftsa.protocols.buildingblocks.ShamirSS import SSS
from ftsa.protocols.buildingblocks.VectorEncoding import VES
from ftsa.protocols.buildingblocks.JoyeLibert import TJLS, ServerKey, EncryptedNumber



class Server(object):
    """
    A server for the FTSA scheme

    ## **Attributes**:
    -------------        
    *step* : `int` --
        The FL step (round).

    *key* : `gmpy2.mpz` --
        The server protection key for TJL

    *ckeys* : `dict` --
        A channel encryption key for each communication channel with each other user {v : key}

    *Y* : `list` --
        The list of all user's protected inputs

    *delta*  : `int` --
        A constant value equals the factorial of nb. of clients
    """

    dimension = 1000 # dimension of the input
    """nb. of elements of the input vector (default: 1000)"""
    valuesize = 16 #-bit input values
    """bit length of each element in the input vector (default: 16)"""
    nclients = 10 # number of FL clients
    """number of FL clients (default: 10)"""
    keysize = 2048 # size of a JL key 
    """size of a TJL key (default: 2048)"""
    threshold = ceil(2*nclients / 3) # threshold for secret sharing
    """threshold for secret sharing scheme (default: 2/3 of the nb. of clients)"""

    # init the building blocks
    VE = VES(keysize // 2, nclients, valuesize, dimension)
    """the vector encoding scheme"""
    TJL = TJLS(nclients, threshold, VE)
    """the threshold JL secure aggregation scheme"""
    pp, _ , _ = TJL.Setup(keysize) # public parameters for TJL
    """the public parameters"""
    prg = PRG(dimension, valuesize)
    """the pseudo-random generator"""
    SS = SSS(PRG.security)
    """the secret sharing scheme"""

    # def __init__(self) -> None:
    def __init__(self, server_key) -> None:
        super().__init__()
        self.step = 0 # the Fl step.
        # self.key = ServerKey(Server.pp, mpz(0)) # the server encryption key for JL (we use zero)
        self.key = server_key
        self.U = [] # set of registered user identifiers
        self.Ualive = [] # set of alive users' identifiers 
        self.Y = [] # aggregation result of the users' ciphertext
        self.delta = 1

    @staticmethod
    def set_scenario(dimension, valuesize, keysize, threshold, nclients, pp):
        """Sets up the parameters of the protocol"""
        Server.dimension = dimension
        Server.valuesize = valuesize
        Server.nclients = nclients
        Server.keysize = keysize
        Server.threshold = threshold
        Server.VE = VES(keysize // 2, 2*(nclients - 1) - 1, valuesize, dimension)
        Server.TJL = TJLS(nclients, threshold, Server.VE)
        Server.TJL.Setup(keysize)
        Server.pp = pp
        Server.prg = PRG(dimension, valuesize)
        Server.SS = SSS(PRG.security)

    def new_fl_step(self, step):
        """Starts a new FL round. 
        
        It increments the round counter and reinitialize the state."""
        self.step = step
        self.Ualive = []
        self.Y = []
        self.delta = 1

    # def setup_register(self, alldhpkc, alldhpks):
    def setup_register(self, alldhpkc):
        """Setup phase - Register: Sever forwards users registrations. 
        
        ** Args **:
        -----------
        *alldhpkc* : `dict`
            The public key of each user (used to construct secret channels)

        *alldhpks* : `dict`
            The public key of each user (used to compute the TJL user keys)

        **Returns**: 
        ----------------
        The same public keys (type: `dict`)
        """
        # assert alldhpkc.keys() == alldhpks.keys()
        assert len(alldhpkc.keys()) >= Server.threshold

        # send for all user public keys
        # return alldhpkc, alldhpks
        return alldhpkc

    def setup_keysetup(self, allekshares):
        """Setup phase - KeySetup: Sever forwards the shares of the TJL keys. 
        
        ** Args **:
        -----------
        *allekshares* : `dict`
            The list of encrypted share generated by each user

        **Returns**: 
        ----------------
        A list of encrypted shares destined to each user (type: `dict`)
        """
        assert len(allekshares) >= Server.threshold

        # prepare eshares for each corresponding user
        ekshares = defaultdict(dict)
        for user in allekshares:
            self.U.append(user)
            for vuser in allekshares[user]:
                ekshares[vuser][user] = allekshares[user][vuser]
        
        self.delta = factorial(len(self.U))

        # send the encrypted key shares for each corresponding user
        return ekshares

    def online_encrypt(self, allebshares, allY):
        """Online phase - Encrypt: Sever forward the shares of the blinding masks and stores the protected inputs. 
        
        ** Args **:
        -----------
        *allebshares* : `dict`
            The list of encrypted share generated by each user

        *allY* : `dict`
            The protected number of each user


        **Returns**: 
        ----------------
        A list of encrypted shares destined to each user (type: `dict`)
        """
        assert len(allebshares) >= Server.threshold

        # prepare eshares for each corresponding user
        ebshares = defaultdict(dict)
        for user in allebshares:
            self.Ualive.append(user)
            for vuser in allebshares[user]:
                ebshares[vuser][user] = allebshares[user][vuser]

        # # aggregate all encrypted messages
        # self.Ytelda = Server.JL.aggregate_vector(list(allY.values()))
        # if len(allY) < len(self.U):
        #     self.Ytelda = [ EncryptedNumber( Server.pp, powmod(x.ciphertext, self.delta, Server.pp.nsquare)) for x in self.Ytelda]
        self.Y = list(allY.values())

        # send the encrypted b shares for each corresponding user
        return ebshares 

    def online_construct(self, allbshares, Yzeroshares = None):
        """Online phase - Construct: Sever construct the blinding masks and the protected zero-value and aggregates the users' inputs. 
        
        ** Args **:
        -----------
        *allbshares* : `dict`
            The list of mask shares of all alive users per user

        *Yzeroshares* : `list`
            A list of shares of the protected zero-value

        **Returns**: 
        ----------------
        The sum of the alive users' inputs (type: `list`)
        """
        assert len(allbshares) >= Server.threshold

        # reconstruct the blinding mask seed b for each user 
        bshares = defaultdict(list)
        for user in allbshares:
            for vuser in allbshares[user]:
                bshares[vuser].append(allbshares[user][vuser])

        lagcoef = []
        b = {}
        B = defaultdict(list)
        for vuser in bshares:
            assert len(bshares[vuser]) >= Server.threshold
            if not lagcoef:
                lagcoef = Server.SS.lagrange(bshares[vuser])
            b[vuser] = Server.SS.recon(bshares[vuser],lagcoef)
            # recompute the blinding vector B
            B[vuser] = Server.prg.eval(b[vuser])
        Yzeroshares = [y for y in Yzeroshares if y]
        if Yzeroshares:
            assert len(Yzeroshares) >= Server.threshold
            # construct the protected zero-value
            Yzero = Server.TJL.ShareCombine(Server.pp, Yzeroshares, self.threshold)
        else:
            Yzero = None
        
        # aggregate
        XplusB = Server.TJL.Agg(Server.pp, self.key, self.step, self.Y, Yzero)

        
        # unmask
        for user in B:
            XplusB = subs_vectors(XplusB, B[user], 2**(Server.VE.elementsize))
        
        return XplusB
