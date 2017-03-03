import libsvm.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

/**
 * Created by HandsomeMrChen on 2017/3/2.
 */
public class Test2 {

    static void run() throws Exception{
         /* 获取训练数据集 */
        Datas traindatas = getDatas(new String("G:/yjs/SVM_weka/src/data/trainSet.txt"));

        /* 定义svm_problem对象 */
        svm_problem problem = new svm_problem();
        problem.l = traindatas.count;
        problem.x = traindatas.datas;
        problem.y = traindatas.lables;

        /* 定义svm_parameter对象*/
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;           //?
        param.kernel_type = svm_parameter.LINEAR;       //?
        param.cache_size = 100;
        param.eps = 0.00001;
        param.C = 1;

        /* 检查参数设置 */
        System.out.println(svm.svm_check_parameter(problem, param));

        /* 训练svm分类模型 */
        svm_model model = svm.svm_train(problem, param);

        /* 获取代预测数据 */
        Datas predictDatas = getDatas(new String("G:/yjs/SVM_weka/src/data/predictSet.txt"));
        int count = 0;
        for(int i = 0; i < predictDatas.count; i++){
            System.out.println(svm.svm_predict(model, predictDatas.datas[i]));
            System.out.println("正确分类为：y=" + predictDatas.lables[i]);
            if(svm.svm_predict(model, predictDatas.datas[i]) == predictDatas.lables[i]){
                count++;
            }
        }
        System.out.println("预测正确率为：" + ((double)count/(double)predictDatas.count));
    }

    /* 从testSet.txt中获取数据点，参数传入数据所在文件 */
    static Datas getDatas(String filename) throws Exception {
        ArrayList<svm_node[]> dataslist = new ArrayList<>();        //存放数据点
        ArrayList<Double> lableslist = new ArrayList<>();           //存放对应的类别
        try(BufferedReader bf = new BufferedReader(new FileReader(filename))){
            String s = "";
            while((s = bf.readLine())!= null){
                String[] content = s.split("\t");
                /* 获取数据节点 */
                svm_node p0 = new svm_node();
                p0.index = 0;
                p0.value = Double.parseDouble(content[0]);
                svm_node p1 = new svm_node();
                p1.index = -1;
                p1.value = Double.parseDouble(content[1]);
                svm_node p[] = {p0, p1};
                dataslist.add(p);

                /* 获取点的类别 */
                lableslist.add(Double.parseDouble(content[2]));
            }
        }

        svm_node[][] datas = new svm_node[dataslist.size()][2];
        double[] lables = new double[lableslist.size()];
        for(int i = 0; i < dataslist.size(); i++){
            datas[i] = dataslist.get(i);
            lables[i] = lableslist.get(i);
        }
        return  new Datas(datas, lables, dataslist.size());
    }
}
