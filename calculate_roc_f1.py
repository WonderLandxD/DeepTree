from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

class Score:
    def __init__(self, y_output, y_label, y_pre):
        self.y_output = y_output
        self.y_label = y_label
        self.y_pre = y_pre
    def cal_roc(self):
        cls = len(self.y_output[0])
        for i in range(cls):
            fpr, tpr, _ = roc_curve(self.y_label, [self.y_output[j][i] for j in range(len(self.y_output))], pos_label=i)
            roc_auc = auc(fpr, tpr)
            print('ok')
            return fpr, tpr, roc_auc        # 只进行一次roc
    def cal_acc(self):
        return accuracy_score(self.y_label, self.y_pre)
    def cal_precision(self):
        # TODO:HER2数据集
        return precision_score(self.y_label, self.y_pre,average='macro')
        # return precision_score(self.y_label, self.y_pre)
    def cal_recall(self):
        # TODO:HER2数据集
        return recall_score(self.y_label, self.y_pre,average='macro')
        # return recall_score(self.y_label, self.y_pre)
    def cal_f1(self):
        # TODO:HER2数据集
        return f1_score(self.y_label, self.y_pre,average='weighted')
        # return f1_score(self.y_label, self.y_pre)

class New_Score:
    def __init__(self, y_label, y_pre):
        self.y_label = y_label
        self.y_pre = y_pre
    def cal_acc(self):
        return accuracy_score(self.y_label, self.y_pre)
    def cal_precision(self):
        # TODO:HER2数据集
        return precision_score(self.y_label, self.y_pre,average='macro')
        # return precision_score(self.y_label, self.y_pre)
    def cal_recall(self):
        # TODO:HER2数据集
        return recall_score(self.y_label, self.y_pre,average='macro')
        # return recall_score(self.y_label, self.y_pre)
    def cal_f1(self):
        # TODO:HER2数据集
        return f1_score(self.y_label, self.y_pre,average='macro')
        # return f1_score(self.y_label, self.y_pre)



# def cal_roc(y_label, y_pre):
#     # y_label=list(np.ravel(_y_label))
#     # y_pre=list(np.ravel(_y_pre))
#     cls = int(len(y_pre)/len(y_label))
#     cls = len(y_pre[0])
#     for i in range(cls):
#         fpr, tpr, _ = roc_curve(y_label, [y_pre[j][i] for j in range(len(y_label))], pos_label=i)
#         roc_auc = auc(fpr, tpr)
#         print('ok')
#         # fig = plt.figure(figsize=(5, 5))
#         # plt.plot(fpr, tpr);
#         # fig.savefig('/data_sdb/dhf/classification-baseline/a3_deit/deit-main/roc_inva.png', dpi=600, format='png')





