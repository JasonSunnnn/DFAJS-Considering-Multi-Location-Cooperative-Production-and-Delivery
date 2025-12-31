% results=[];
% repeat=10;
% for i=1:15
%     results1=[];
%     for ii=1:repeat
%         results1=[results1;results_3HSEA{i,ii}];
%         results1=[results1;results_3HSEA_IP{i,ii}];
%         results1=[results1;results_3HSEA_CP{i,ii}];
%         results1=[results1;results_3HSEATS{i,ii}];
%         results1=[results1;results_3NSGAII{i,ii}];
%         results1=[results1;results_3NSGAIII{i,ii}];
%         results1=[results1;results_3NSGAIIITS{i,ii}];
%     end
%     results{i}=results1;
% end
clear
load('ReferenceSet.mat') 
load('Algorithm results .mat')
%%
repeat=10;
HVcell=[];
HVmean=zeros(15,3);
for i=1:15
    i
    min1=min(results{i});
    max1=max(results{i});
    for ii=1:repeat %重复次数
        d=max1-min1;
        d(d==0)=5;
        
        temp=results_3MMOEA{i,ii}.Objv;
        temp=(temp-min1)./d;
        HV_MMOEA(i,ii)= P_evaluate('HV',temp,[1.1,1.1,1.1]);
        
        temp=results_3HEDA{i,ii}.Objv;
        temp=(temp-min1)./d;
        HV_HEDA(i,ii)= P_evaluate('HV',temp,[1.1,1.1,1.1]);
        
        
%         HV_MMOEA(i,ii)= P_evaluate('HV',temp,[1.1,1.1,1.1]);
%         
%         temp=results_3HEDA{i,ii};
%         temp=(temp-min1)./d;
%         HV_HEDA(i,ii)= P_evaluate('HV',temp,[1.1,1.1,1.1]);
%         
%         temp=results_3DQNMA{i,ii};
%         temp=(temp-min1)./d;
%         HV_DQNMA(i,ii)= P_evaluate('HV',temp,[1.1,1.1,1.1]);
%         
%         temp=results_3HSEATS{i,ii};
%         temp=(temp-min1)./d;
%         HV_NSFATS(i,ii)= P_evaluate('HV',temp,[1.1,1.1,1.1]);
    end
end
HVmean(:,1)=mean(HV_MMOEA')';
HVmean(:,2)=mean(HV_HEDA')';

% HVcell{1}=HV_MMOEA;
% HVcell{2}=HV_HEDA;
% HVcell{3}=HV_DQNMA;
% HVcell{4}=HV_NSFATS;


% HVmean(:,4)=mean(HV_3HSEATS1')';
% HVmean(:,5)=mean(HV_NSGAIII')';
% HVmean(:,6)=mean(HV_NSGAII')';

%%
objv1=zeros(repeat,3);
objv2=objv1;
objv11=cell(1);
objv22=cell(1);
objv33=objv22;
objv=cell(1);
for i=1:15
    objv1=[];
    objv2=[];
    objv3=[];
    objv4=[];
    for ii=1:repeat
        objv1=[objv1;min([results_3HSEA_IP{i,ii};results_3HSEA_IP{i,ii}])];
        objv2=[objv2;min([results_3HSEA_CP{i,ii};results_3HSEA_CP{i,ii}])];
        objv3=[objv3;min([results_3HSEA{i,ii};results_3HSEA{i,ii}])];
        objv4=[objv4;min([results_3HSEA_CP{i,ii};results_3HSEA_CP{i,ii}])];
    end
    objv11{i,1}=objv1(:,1);
    objv11{i,2}=objv2(:,1);
    objv11{i,3}=objv3(:,1);
    
    objv22{i,1}=objv1(:,2);
    objv22{i,2}=objv2(:,2);
    objv22{i,3}=objv3(:,2);
    
    objv33{i,1}=objv1(:,3);
    objv33{i,2}=objv2(:,3);
    objv33{i,3}=objv3(:,3);
end

%%
HV_HSEA=[];
HV_HSEA_IP=[];
HVmean=zeros(15,1);
for i=1:15
    min1=min(results{i});
    max1=max(results{i});
    for ii=1:10 %重复次数
        %     temp=results_2encode{i};
        %     temp=(temp-min(results))./(max(results)-min(results));
        %     HV1(i)=P_evaluate('HV',temp,[1.1,1.1,1.1]);
        d=max1-min1;
        d(d==0)=5;
        temp=results_3HSEA{i,ii};
        temp=(temp-min1)./d;
        HV_HSEA(i,ii)= P_evaluate('HV',temp,[1.1,1.1,1.1]);
        
    end
end
HVmean(:,1)=mean(HV_HSEA')';


%%
runtime_NSGAII=cell2mat(runtime_3MMOEA);
runtime_NSGAIII=cell2mat(runtime_3HEDA);
runtime_HSEA=cell2mat(runtime_3DQNMA);
runtime_HSEATS=cell2mat(runtime_3HSEATS);

Runtime=cell(1);
Runtime{1}=runtime_NSGAII;
Runtime{2}=runtime_NSGAIII;
Runtime{3}=runtime_HSEA;
Runtime{4}=runtime_HSEATS;


% Runtime=zeros(15,1);
% runtime_NSGAII=cell2mat(runtime_NSGAII);
% Runtime(:,1)=mean(runtime_3DQNMA')';
% runtime_3HSEA=cell2mat(runtime_3HSEA);
% Runtime(:,2)=mean(runtime_3HSEA')';
% runtime_3NSGAIII=cell2mat(runtime_3NSGAIII);
% Runtime(:,3)=mean(runtime_3NSGAIII')';
% runtime_3NSGAII=cell2mat(runtime_3NSGAII);
% Runtime(:,4)=mean(runtime_3NSGAII')';


%% 秩和检验
RankSumTest=zeros(15,3);
for i=1:15
    %reference_col=HVcell{4}(i,:);
    reference_col=runtime_3NSFATS(i,:);
    %reference_col = objv33{i,3};
    for ii=1:1
        data=runtime_3HSEA(i,:);
        %data=objv33{i,ii};
        % 对每一列与最后一列进行Wilcoxon秩和检验
        [p, h] = ranksum(data, reference_col);
        
        % 判断显著性
        if h == 1
            % 比较中位数
            if median(data) > median(reference_col)
                RankSumTest(i,ii) = 1; % 显著大于
            else
                RankSumTest(i,ii) = 2; % 显著小于
            end
        else
            RankSumTest(i,ii) = 0; % 没有显著差异
        end
    end
end
%% ANOVA 
% % 0. 数据准备（示例：3 组，30 次独立运行）
% X = [HV_NSFATS', HV_HEDA', HV_DQNMA'];   % 每列一组，行 = 独立运行
% 
% % 1. 正态性检验（Jarque-Bera，自带）
% normOK = true;  pNorm = zeros(1,g);
% for gg = 1:g
%     [~, pNorm(gg)] = jbtest(X(:,gg), alpha);   % 自带函数
%     if pNorm(gg) < alpha, normOK = false; end
% end
% fprintf('Jarque-Bera p-values: %s\n', mat2str(pNorm'));
% if ~normOK, warning('Normality violated → use non-parametric test'); end
% 
% % 2. 方差齐性检验（Levene）
% % 假设X和Y是两个样本数据
% [h, p] = vartestn(X, 'Centered', 'off');
% 
% if h == 0
%     disp('Homogeneity of Variance，p value = ', p);
% else
%     disp('Heterogeneity of Variance，p value = ', p);
% end
% 
% 
% % 3. 非参检验（不满足 ANOVA 假设时）
% if g == 2
%     % 两组：Wilcoxon rank-sum（双侧）
%     [pNon, hNon, statsNon] = ranksum(X(:,1), X(:,2), 'Alpha', alpha, 'Tail', 'both');
%     fprintf('Wilcoxon rank-sum p = %.4f, h = %d\n', pNon, hNon);
% else
%     % 多组：Kruskal-Wallis
%     [pNon, tblNon, statsNon] = kruskalwallis(X, [], 'off');
%     fprintf('Kruskal-Wallis p = %.4f\n', pNon);
% end
% 
% % 4. 结论
% if pNon < alpha
%     fprintf('→ Significant difference at α = %.2f\n', alpha);
% else
%     fprintf('→ No significant difference at α = %.2f\n', alpha);

% end
