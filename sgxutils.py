from torch.utils.cpp_extension import load

class SGXUtils():
    def __init__(self):
        self.lib = load(
            name="sgx", 
            sources=["App/pytorch_extension.cpp"],
        )

        self.lib.init_sgx()

    # given a weight & precompute w*r
    def precompute(self, w, batch):
        self.lib.precompute(w.T.cpu(), batch)

    # given a input vector, compute inp + r
    def addNoise(self, inp):
        return self.lib.addNoise(inp.cpu()).cuda()

    # recover the results using w*r
    def removeNoise(self, inp):
        return self.lib.removeNoise(inp.cpu()).cuda()

    # run matrix multiplicaiton inside sgx Encalve
    def nativeMatMul(self, w, inp):
        return self.lib.nativeMatMul(w.T.cpu(), inp.cpu())