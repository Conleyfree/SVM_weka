/* creator: czc
   time: 2017.2.28 15:36
   reason: svm replys on Weka.
 */

import libsvm.*;

public class Main {

    public static void main(String[] args) throws Exception{

//      Test1.run();     /* 初涉libsvm */
//      Test2.run();     /* 数量较少，维度只有2的数据集 */
//      Test3.run();     /* 数量多很多，维度为4的数据集 */
//      DemoTest.run();
        Test4.run();     /* 基于聚类分析的SVM */

    }

}

class Datas{
    svm_node[][] datas;
    double[]     lables;
    int count;
    Datas(svm_node[][] datas, double[] lables, int count){
        this.datas = datas;
        this.lables = lables;
        this.count = count;
    }
}


/* 注：
 * svm_node表示的是{向量的分量序号，向量的分量值}，很多稀疏矩阵均用此方法存储数据，可以节约空间；
 * svm_node[]则表示一个向量，一个向量的最后一个分量的svm_node.index用-1表示；
 * svm_node[][]则表示一组向量，也就是训练集。
 */
