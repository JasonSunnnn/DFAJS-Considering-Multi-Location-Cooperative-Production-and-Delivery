%画箱线图
close all
insname={'INS-0501','INS-0502','INS-0503','INS-0504',...
    'INS-0505','INS-1001','INS-1002','INS-1003','INS-1004','INS-1005','INS-3001','INS-3002','INS-3003','INS-3004','INS-3005'};
objname={'Cmax','DT','Cost'};
for i=[1 6 11]
    for ii=1:3
        % 生成三组随机数据
        a1=[];a2=[];a3=[];a4=[];a5=[];a6=[];a7=[];a8=[];
        for iii=1:10
            minV=min([results_3NSGAII{i,iii}.Objv;results_3NSGAII{i,iii}.Objv]);
            a1=[a1;minV(ii)];
            minV=min([results_3MOEAD{i,iii}.Objv;results_3MOEAD{i,iii}.Objv]);
            a2=[a2;minV(ii)];
            minV=min([results_3NSGAIII{i,iii};results_3NSGAIII{i,iii}]);
            a3=[a3;minV(ii)];
            minV=min([results_3NSFA{i,iii};results_3NSFA{i,iii}]);
            a4=[a4;minV(ii)];
            minV=min([results_3MMOEA{i,iii}.Objv;results_3MMOEA{i,iii}.Objv]);
            a5=[a5;minV(ii)];
            minV=min([results_3HEDA{i,iii}.Objv;results_3HEDA{i,iii}.Objv]);
            a6=[a6;minV(ii)];
            minV=min([results_3DQNMA{i,iii};results_3DQNMA{i,iii}]);
            a7=[a7;minV(ii)];
            minV=min([results_3NSFATS{i,iii};results_3NSFATS{i,iii}]);
            a8=[a8;minV(ii)];
        end
        
        % 将数据组合成一个矩阵
        data = [a1,a2,a3,a4,a5,a6,a7,a8];
        data=data(:,[1,2,3,4,5,6,7,8]);
            
        % 绘制箱线图
        figure
        boxplot(data, 'Labels', {'NSGA-II','MOEA/D','NSGA-III','NSFA','MMOEA','HEDA-DEV','DQNMA','NSFATS'});
        ylabel(objname{ii});
        title(insname{i});
        ax = gca;

        set(ax.Title, 'FontSize', 12);
        % 设置 x 轴标签的字号
        set(ax.XAxis, 'FontSize', 10);
        % 设置 y 轴标签的字号
        set(ax.YAxis, 'FontSize', 12);
        set(gca,'ytick',[],'yticklabel',[])
    end
end