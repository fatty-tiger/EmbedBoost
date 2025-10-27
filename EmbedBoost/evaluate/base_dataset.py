from abc import ABC, abstractmethod


class AbsRetrievalEvalDataset(ABC):
    @abstractmethod
    def load_querys(self):
        pass

    @abstractmethod
    def load_corpus(self):
        pass
