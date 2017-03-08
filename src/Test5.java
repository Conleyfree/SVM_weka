import libsvm.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.LinkedList;

/**
 * Created by HandsomeMrChen on 2017/3/3.
 */
public class Test5 {

    private static ArrayList<Double[]> standarddata = new ArrayList<>();
    private static ArrayList<Double[]> cluster0 = new ArrayList<>();
    private static ArrayList<Double[]> cluster1 = new ArrayList<>();

    private static Datas traindatas;       //训练集
    private static Datas predictdatas;     //测试集

    static void run() {
        standardization("G:/yjs/SVM_weka/src/data/cluster0.csv");
        k_means();
        getDate();

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
        System.out.println(svm.svm_check_parameter(problem, param));    //如果参数没有问题，则svm.svm_check_parameter()函数返回null,否则返回error描述。

        svm_model model = svm.svm_train(problem, param);

        predict(model);
    }

    /* 数据规格化处理 */
    private static void standardization(String filename){
        ArrayList<Double> min = new ArrayList<>();
        ArrayList<Double> max = new ArrayList<>();

        /* 把数据从文件中读取出来保存到standarddata中，并找出各个数据项的最大值与最小值 */
        try(BufferedReader bf = new BufferedReader(new FileReader(filename))){
            String s = "";
            s = bf.readLine();
            String[] contents = s.split(",");
            Double[] record = new Double[contents.length];
            for(int i = 0; i < contents.length; i++){
                Double d = Double.parseDouble(contents[i]);
                min.add(d);
                max.add(d);
                record[i] = d;
            }
            standarddata.add(record.clone());       //加入一条记录
            while((s=bf.readLine())!=null){
                contents = s.split(",");
                for(int i = 0; i < contents.length; i++){
                    Double d = Double.parseDouble(contents[i]);
                    if(d < min.get(i)){
                        min.set(i, d);
                    }else if(d > max.get(i)){
                        max.set(i, d);
                    }
                    record[i] = d;
                }
                standarddata.add(record.clone());       //加入一条记录
            }
        }catch(Exception ex){
            ex.printStackTrace();
        }

        /* 归一化计算 */
        for(int i = 0; i < standarddata.size(); i++){
            Double[] record = standarddata.get(i);
            for(int j = 0; j <record.length; j++){
                record[j] = (record[j] - min.get(j)) / (max.get(j) - min.get(j));
            }
            standarddata.set(i, record);
        }
    }

    /* k-means 算法 */
    private static void k_means(){
        Double[][] m = initialCenter(standarddata, 2);
        Double[] m0 = m[0];
        Double[] m1 = m[1];

//        Double[] m0 = standarddata.get(0);
//        Double[] m1 = standarddata.get(1);

        Double[] new_m0;
        Double[] new_m1;

        divide(standarddata, m0, m1);   //把归一化数据集聚类成两个初始簇

        do{
            new_m0 = GetCenter(cluster0, m0.length);
            new_m1 = GetCenter(cluster1, m1.length);
            if(isEqual(new_m0, m0) && isEqual(new_m1, m1)){  //两个簇的中心已经确定下来了
                break;
            }else{
                m0 = new_m0;
                m1 = new_m1;
                cluster0.clear();   //清空集合
                cluster1.clear();   //清空集合
                divide(standarddata, m0, m1);
            }
        }while(true);

    }

    /* 计算点之间的欧式距离,为了计算方便省去最后的开方计算 */
    private static double distance(Double[] p1, Double[] p2){
        double result = 0;
        for(int i = 0; i < p1.length; i++){
            result += (p1[i] - p2[i]) * (p1[i] - p2[i]) ;
        }
        return  result;
    }

    /* 计算簇的中心, 参数cluster：簇集合；参数length：记录的属性个数*/
    private static Double[] GetCenter(ArrayList<Double[]> cluster, int length){
        Double[] center = new Double[length];
        if(!cluster.isEmpty()){     //非空
            for(int i = 0; i < length; i++){       //初始化数组
                center[i] = 0.0;
            }
            for(Double[] record : cluster){
                for(int i = 0; i < length; i++){
                    center[i] += record[i];
                }
            }
            for(int i = 0; i < length; i++){
                center[i] = center[i] / cluster.size();
            }
            return  center;
        }else{
            return null;
        }
    }

    /* 比较两个点是否相同 */
    static Boolean isEqual(Double[] p1, Double[] p2){
        if(p1.length != p2.length )     return false;
        for(int i = 0; i < p1.length; i++){
            if(p1[i].doubleValue() != p2[i].doubleValue())  return false;      //注意：Double是类
        }
        return  true;
    }

    /* 聚类 */
    static void divide(ArrayList<Double[]> datas, Double[] m0, Double[] m1){
        for(Double[] p : datas){         //把归一化数据集聚类成两个初始簇
            if(distance(p, m0) < distance(p, m1)){
                cluster0.add(p);
            }else{
                cluster1.add(p);
            }
        }
    }

    /* 获取数据集 */
    private static void getDate(){
        LinkedList<svm_node[]> trains = new LinkedList<>();
        LinkedList<svm_node[]> predicts = new LinkedList<>();
        LinkedList<Double> trainslables = new LinkedList<>();
        LinkedList<Double> predictslables = new LinkedList<>();

        /* 分别从两个簇中取出70%放入训练集，30%放入测试集 */
        int count = 1;
        for(Double[] d : cluster0){
            svm_node[] p = creatNodeVector(d);
            if(count / (double)cluster0.size() < 0.7) {
                trains.add(p);
                trainslables.add(1.0);
            } else {
                predicts.add(p);
                predictslables.add(1.0);
            }
            count++;
        }
        count = 1;
        for(Double[] d : cluster1){
            svm_node[] p = creatNodeVector(d);
            if(count / (double)cluster1.size() < 0.7) {
                trains.add(p);
                trainslables.add(-1.0);
            } else {
                predicts.add(p);
                predictslables.add(-1.0);
            }
            count++;
        }

        /* 生成训练数据集 */
        svm_node[][] traSet = new svm_node[trains.size()][1];
        double[] tralables = new double[trainslables.size()];
        for(int i = 0; i < trains.size(); i++){
            traSet[i] = trains.get(i);
            tralables[i] = trainslables.get(i);
        }
        traindatas = new Datas(traSet, tralables, trains.size());

        /* 生成预测数据集 */
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

    /* 创建svm_node[]向量 */
    private static svm_node[] creatNodeVector(Double[] d){
        svm_node[] p = new svm_node[d.length];
        for(int i = 0; i < d.length; i++){      //创建SVM数据向量
            if(i == d.length - 1)   p[i] = creatSVMNode(-1, d[i]);
            else                    p[i] = creatSVMNode(i, d[i]);
        }
        return p;
    }

    /* 初始化簇中心 */
    private static Double[][] initialCenter(ArrayList<Double[]> datas, int n){
        int v = datas.get(0).length;        //维度
        Double[][] m = new Double[n][v];
        for(int i = 0; i < n; i++) {         //先按中心循环
            int count = 0;
            for (int j = 0; j < v; j++) {    //初始化数组内容
                m[i][j] = 0.0;
            }

            for (int k = i + n * count; k < datas.size(); ++count, k = i + n * count) {
                for (int j = 0; j < v; j++) {
                    Double[] record = datas.get(k);
                    m[i][j] += record[j];    //同属性值相加
                }
            }
            for (int j = 0; j < v; j++) {
                m[i][j] /= (double) count;   //求出相同属性下的均值
            }
        }
        return  m;
    }

    /* 运行测试并打印 */
    private static void predict(svm_model model){
        double num = 0;
        for(int i = 0; i < predictdatas.count; i++){
            double result = svm.svm_predict(model, predictdatas.datas[i]);
            System.out.println("预测结果属于:" + (result==1?"簇0":"簇1") + "; 正确结果：" + (predictdatas.lables[i]==1?"簇0":"簇1"));
            if(result == predictdatas.lables[i])   num++;
        }
        System.out.println("预测正确率为：" + num/predictdatas.count);
    }
}
