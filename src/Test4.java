import libsvm.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.LinkedList;

/**
 * 先对数据进行预处理，对数据进行聚类分析，再取出80%的数据作为SVM训练数据集，另外20%作为测试数据集检测SVM分类效果。
 * Created by HandsomeMrChen on 2017/3/3.
 */
public class Test4 {

    private static ArrayList<Double> standarddata = new ArrayList<>();
    private static ArrayList<Double> cluster0 = new ArrayList<>();
    private static ArrayList<Double> cluster1 = new ArrayList<>();
    private static int max_index = -1;      //标记数据最大值位置
    private static int min_index = -1;

    private static Datas traindatas;       //训练集
    private static Datas predictdatas;     //测试集

    static void run(){
        standardization("G:/yjs/SVM_weka/src/data/4200.csv");
        k_means();
        getDate();

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

        svm_model model = svm.svm_train(problem, param);

        double num = 0;
        for(int i = 0; i < predictdatas.count; i++){
            double result = svm.svm_predict(model, predictdatas.datas[i]);
            System.out.println("预测结果属于" + (result==1?"簇0":"簇1") + "; 正确结果：" + (predictdatas.lables[i]==1?"簇0":"簇1"));
            if(result == predictdatas.lables[i])   num++;
        }
        System.out.println("预测正确率为：" + num/predictdatas.count);

    }

    /* 数据规格化处理 */
    private static void standardization(String filename){
        double max = 0, min = Integer.MAX_VALUE;
        try(BufferedReader bf = new BufferedReader(new FileReader(filename))){
            String s = "";
            int count = 0;
            while((s=bf.readLine())!=null){
                double d = Double.parseDouble(s);
                if (d<min){
                    min = d;    min_index = count;
                }
                if (d>max){
                    max = d;    max_index = count;
                }
                standarddata.add(d);
                count++;
            }
        }catch(Exception ex){
            System.out.println(ex);
        }
        for(int i = 0; i < standarddata.size(); i++){
            standarddata.set(i, (standarddata.get(i)-min)/(max-min));
        }
    }

    /* K-means算法 */
    private static void k_means(){
        double m0 = standarddata.get(max_index), m1 = standarddata.get(min_index);      //初始分别以最大值和最小值作为簇的中心
        double new_m0 = m0, new_m1 = m1;
        do{
            m0 = new_m0;
            m1 = new_m1;
            cluster0.clear();
            cluster1.clear();
            for(int i = 0; i < standarddata.size(); i++){
                Double d = standarddata.get(i);
                if(Math.abs(d-m0) < Math.abs(d-m1)){
                    cluster0.add(d);
                }else{
                    cluster1.add(d);
                }
            }
            new_m0 = getMid(cluster0);
            new_m1 = getMid(cluster1);
        }while(new_m0 != m0 || new_m1 != m1);
//        m0 = new_m0;
//        m1 = new_m1;
//        System.out.println("簇0的中心：" + m0);
//        for(Double d : cluster0){
//            System.out.println(d);
//        }
//        System.out.println("簇0的个数有：" + cluster0.size() );
//        System.out.println("簇1的中心：" + m1);
//        for(Double d : cluster1){
//            System.out.println(d);
//        }
//        System.out.println("簇1的个数有：" + cluster1.size() );
    }

    /* 获得簇中心 */
    private static double getMid(ArrayList<Double> cluster){
        Iterator<Double> it = cluster.iterator();
        double middle = 0;
        for(;it.hasNext();){
            middle += it.next();
        }
        return middle/cluster.size();
    }

    /* 获取数据集 */
    private static void getDate(){
        LinkedList<svm_node[]> trains = new LinkedList<>();
        LinkedList<svm_node[]> predicts = new LinkedList<>();
        LinkedList<Double> trainslables = new LinkedList<>();
        LinkedList<Double> predictslables = new LinkedList<>();

        /* 分别从两个簇中取出80%放入训练集，20%放入测试集 */
        int count = 1;
        for(Double d : cluster0){
            svm_node p0 = creatSVMNode(-1, d);
            svm_node[] p = {p0};
            if(count / (double)cluster0.size() < 0.8) {
                trains.add(p);
                trainslables.add(1.0);
            } else {
                predicts.add(p);
                predictslables.add(1.0);
            }
            count++;
        }
        count = 1;
        for(Double d : cluster1){
            svm_node p0 = creatSVMNode(-1, d);
            svm_node[] p = {p0};
            if((count / (double)cluster1.size()) < 0.8) {
                trains.add(p);
                trainslables.add(-1.0);
            } else {
                predicts.add(p);
                predictslables.add(-1.0);
            }
            count++;
        }

        svm_node[][] traSet = new svm_node[trains.size()][1];
        double[] tralables = new double[trainslables.size()];
        for(int i = 0; i < trains.size(); i++){
            traSet[i] = trains.get(i);
            tralables[i] = trainslables.get(i);
        }
        traindatas = new Datas(traSet, tralables, trains.size());

        svm_node[][] preSet = new svm_node[predicts.size()][1];
        double[] prelables = new double[predictslables.size()];
        for(int i = 0; i < predicts.size(); i++){
            preSet[i] = predicts.get(i);
            prelables[i] = predictslables.get(i);
        }
        predictdatas = new Datas(preSet, prelables, predicts.size());
    }

    /* 创建svm_node */
    private static svm_node creatSVMNode(int index, double value){
        svm_node p = new svm_node();
        p.index = index;
        p.value = value;
        return p;
    }
}
