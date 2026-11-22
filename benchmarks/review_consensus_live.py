"""Freshly phrased real requests using deployed consensus configuration."""
from .review_guarded_dtm import main,ROOT
from .core import read

REQUESTS=[
 ('implicit_ko','클릭이나 구매 같은 암묵적 피드백으로 개인화된 추천 순위를 학습하는 방법을 찾고 싶어.'),
 ('cross_domain_en','Methods for recommendation across domains that transfer knowledge to alleviate sparse user item feedback.'),
 ('mobile_ux_en','Experience sampling methods for studying mobile user experience in everyday life, supported by case studies.'),
 ('fpga_en','FPGA implementations of neural network inference that report power consumption and throughput. Exclude surveys.'),
 ('sgd_en','Methods that give convergence guarantees for stochastic gradient descent in nonconvex optimization.'),
 ('node_en','Graph neural networks for classifying nodes by aggregating attributes from neighboring nodes.'),
]

if __name__=='__main__':
    selected=read(ROOT/'output/balanced_benchmark_20261117/route_consensus/selected.json')
    main('live',True,requests=REQUESTS,output_name='consensus_live_review',consensus_strength=selected['strength'])
