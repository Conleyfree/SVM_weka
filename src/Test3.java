import libsvm.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.LinkedList;

/** 使用了更多维度且量更多的数据，数据用.csv文件保存
 * Created by HandsomeMrChen on 2017/3/2.
 */
public class Test3 {

    static Datas traindatas;       //训练集
    static Datas predictdatas;     //测试集

    static void run() throws Exception{

        getDatas("G:/yjs/SVM_weka/src/data/invest.csv");

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
        double num = 0;
        for(int i = 0; i < predictdatas.count; i++){
            double result = svm.svm_predict(model, predictdatas.datas[i]);
            System.out.println("预测结果为：" + (result==1?"yes":"no") + "; 正式结果：" + (predictdatas.lables[i]==1?"yes":"no"));
            if(result == predictdatas.lables[i])   num++;
        }
        System.out.println("预测正确率为：" + num/predictdatas.count);
    }

    /* 从invest.csv文件中获取数据集 */
    static void getDatas(String filename) throws Exception{
        LinkedList<svm_node[]> trains = new LinkedList<>();
        LinkedList<svm_node[]> predicts = new LinkedList<>();
        LinkedList<Double> trainslables = new LinkedList<>();
        LinkedList<Double> predictslables = new LinkedList<>();
        try(BufferedReader bf = new BufferedReader(new FileReader(filename))) {
            String s = "";
            int i = 1;
            while((s = bf.readLine()) != null){
                String[] contents = s.split(",");
                svm_node[] p = {creatSVMNode(0, transform(contents[1])), creatSVMNode(1, transform(contents[2])), creatSVMNode(2, transform(contents[3])), creatSVMNode(-1, transform(contents[4]))};
                double lable = transform(contents[5]);
                if(i%10 == 0){      //取数据集 1/10 作为预测数据集
                    predicts.add(p);
                    predictslables.add(lable);
                }else{
                    trains.add(p);
                    trainslables.add(lable);
                }
                i++;
            }
        }
        svm_node[][] traSet = new svm_node[trains.size()][4];
        double[] tralables = new double[trainslables.size()];
        for(int i = 0; i < trains.size(); i++){
            traSet[i] = trains.get(i);
            tralables[i] = trainslables.get(i);
        }
        traindatas = new Datas(traSet, tralables, trains.size());

        svm_node[][] preSet = new svm_node[predicts.size()][4];
        double[] prelables = new double[predictslables.size()];
        for(int i = 0; i < predicts.size(); i++){
            preSet[i] = predicts.get(i);
            prelables[i] = predictslables.get(i);
        }
        predictdatas = new Datas(preSet, prelables, predicts.size());
    }

    /* 为invest.csv 设置的“字符串/值”的转换 */
    static double transform(String s){
        switch (s){
            case "yes" : return 1.0;
            case "no" : return  -1.0;
            case "high" : return 1.0;
            case "low" : return -1.0;
            default: return 0;
        }
    }

    /* 创建svm_node */
    static svm_node creatSVMNode(int index, double value){
        svm_node p = new svm_node();
        p.index = index;
        p.value = value;
        return p;
    }
}
