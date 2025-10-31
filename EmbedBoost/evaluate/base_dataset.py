from abc import ABC, abstractmethod


class AbsRetrievalEvalDataset(ABC):
    @abstractmethod
    def load_datas(self):
        pass
