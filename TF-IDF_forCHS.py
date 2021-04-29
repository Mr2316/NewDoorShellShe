#coding=utf8

import numpy as np 
from collections import Counter
import itertools
import jieba


np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

answers = [
	"应聘人员经面试评审程序最终被录用后，会收到人力资源处的“ 录用通知”。被录用员工应在规定期限内亲自办理报到手续，并提交相关材料。如有特殊情况不能在规定的时间报到，应事先与人力资源处和所聘用部门说明原因。不同人员类别如国内应届毕业Th、有工作经历人员、海外人员、港澳台地区人员所提  交的材料各不相同，详见宁波材料所人力资源处办事指南专栏。提交虚假材料，视为严重违反诚信的行为，可作为被辞退的理由。",
	"所有正式办理报到手续的员工，均应在入职一个月内签订聘用合同，合同类型主要分以下三类：\n（一）岗位聘用合同：适用于事业编制高层次员工，主要面向所聘岗位需要较长时限内持续工作的员工，如科研部门副高及以上和管理部门副主任及以上人员，聘期原则上4年，试用期6个月，按干部任命者试用期12个月；\n（二）任期聘用合同：主要面向科研部门的中级及管理部门的主管人员，聘期原则上3年，试用期6个月。\n（三）项目聘用合同：主要面向初级及其他各类员工，聘期原则上3年，试用期6个月。\n试用期包含在合同期内。试用期间，部门或团队负责人指定专人“引导员”帮助新员工。引导员的职责包括向新员工介绍本部门基本情况，帮助了解单位有关规则和规定，确认并协助领取《员工手册》等资料。任何有关工作的具体事务， 如领取办公用品、设备使用、纪律遵守、用餐等，引导员都有义务对新员工进行介绍。",
	"人事档案是对员工各方面情况的真实客观的记录，对人事档案进行恰当的管理十分重要，是对每位员工的尊重。",
	"员工本人持人力资源处开具的《人员商调函》及本人填写好的《人员调动（流动）登记表》回原档案所在地办理档案调动和人事关系转移手续；离职后一个月内请将档案及时转出。",
	"为提高员工的知识技能及发挥其潜在智能，使人力资源能适应本院日益迅速发展的需要，宁波材料所将举行各种教育培训活动，被指定员工，不得无故缺席，确有特殊原因，应按有关请假制度执行。每位员工均应该参加与宁波材料所业务有关的培训课程，并建立培训记录。这些记录将作为对员工的工作能力评估的一部分。任何员工均享有新员工培训及岗位技能培训的权利及义务；均可以利用工作及业余时间参加宁波材料所组织的在职公共课程的培训，培训时间、地点及参与人员详见培训组织部门下发的通知。",
	"合同期满前，按照分级管理原则，分别由相应的部门或人力资源处组织合同期满考核。其中部门负责人和团队负责人由宁波材料所组织续聘考核。其他员工由所在部门组织续聘考核。聘期期满考核程序如下：个人任期述职，评委综合评议并根据聘期表现给出是否续聘、是否低聘或者解聘的建议。结果使用： 人力资源处根据考核结果对人员进行解聘或续聘，必要时与工资、奖金等福利挂钩。",
	"（一）员工持部门及团队负责人签批的同意辞职报告（不含被辞退者），到人力资源处领取《离职转单》；（二） 离所手续办理完毕后， 人力资源处给员工本人出具《解除（终止）劳动合同证明》；（三） 办理社会保险转移手续： 一般是在停薪后下个月的18日至当月月底办理在支付宝--城市服务—社保中办理；（四）办理人事档案转移手续：个人需持新单位或档案托管人才中心出具的《调档函》及我单位出具的《解除（终止）劳动合同证明》到宁波市人才交流和服务中心办理调动手续；宁波市人才交流服务中心地址：兴宁东路228号。（五）集体户口的迁移：集体户口的员工，需在解除或终止劳动关系后的30个工作日内办理集体户口的迁出手续，不及时迁出户口的，按滞留月份每月收取10元管理费。",
	"工作时间是星期一至星期五，每天8小时。其中： 上班时间：8:30—17:30；午休时间：12:30开始休一个小时，13:30到岗。",
	"（一）执行日考勤制度，月考勤周期为当月。如果不能按时上班，员工有义务在所内网考勤系统请假，预先通知直属上级， 说明缺勤理由和预计上班时间。如不事先请假（紧急情况也需通过短信、微信、邮件等有效方式告知直属上级） 并获批准即缺勤，则视为旷工。（二）全所实行上下班打卡考勤制度，每位员工上下班时必须自觉打卡，严禁他人代替打卡考勤。（三）员工在1个考勤月中迟到或早退3 次，扣除1 天绩效工资；旷工1天停发1个月工资；连续旷工10个工作日或1年内累计旷工20个工作日，按自动离职处理。",
	"（一）请假手续实行层级负责制。一般员工请假，须由所在团队负责人批准，报人力资源处备案；各部门负责人请假，报主管所领导审批；所领导个人请假，按干部管理权限和规定办理。前述休假制度中有特殊规定的，按相关规定办理。（二）员工享受相应的假期须办理请假手续，首先应到人力资源处领取并填写请假表，交所在部门负责人或主管院领导审批签字，再交人力资源处审核备案。（三）员工假期满后应及时返回单位工作，且在上班当天按批准权限办理销假手续。如不及时销假，其超过的时间按事假累积计算。（四）员工如无特殊情况，必须按照本条（一）、（二）规定事先办理请假审批手续。未按规定办理请假手续的，或请假期满逾期不归、又未履行续假手续的，按旷工处理。关于员工考勤与休假制度的具体信息以所内网规章制度为准。",
	"凡正式员工均可享受一定年限和额度的租房补贴，具体以内网发布的相关规章制度为准。",
	"正式员工子女入托儿所或幼儿园后， 享受入托费用补贴，每位员工限一个小孩，标准根据相关文件执行，每年年底集中办理一次.",
	"高温补贴发放月数4个月（6月、7月、8月、9月）， 发放标准参照浙江省劳动和社会保障局关于高温费发放标准执行。",
	"所有正式员工享受定额午餐补贴，发放至餐卡中。",
	"一定级别的员工，根据工作需要，可享受通讯补贴。",
	"在端午节、中秋节、春节等中国传统节日，所有员工均享受一定礼品慰问，由工会确定。",
	"员工可登陆支付宝城市服务中， 查询个人的缴费和记账情况。或拨打劳动保障热线12333就与社保相关的问题进行电话咨询。",
	"女职工流产、生育，男职工计划生育手术或男职工未就业配偶生育者，凡符合宁波市生育保险金办理条件的均可领取生育保险金。需要注意的是在流产、生育前请及时联系人力资源处，以便为您及时办理相关手续，告知注意事项。",
	"员工可登陆支付宝城市服务查询，或拨打电话12329查询个人账户余额和缴费明细。",
	"单位和个人缴费比例均为12%，全部计入个人账户。",
	"员工租房、购房、自建、翻建、大修自住住房者；退休及调离本市者；失业人员等凡符合相关规定者均可提取公积金，可在支付宝城市服务中申请提取",
	"所（局）级领导干部从事兼职活动，须向中科院报备审批；科研人员从事兼职活动，需根据兼职机构性质分别报技术转移、科技综合处审批。全职承担战略性先导科技专项任务的骨干人员须报先导专项领导小组和总体组（部）审核备案。"
]

keywords = [
	["录用"],
	["合同","试用期"],
	["人事","档案"],
	["调","转","档案","调档"],
	["员工","培训"],
	["聘期","考核","合同","期满"],
	["离职","手续"],
	["工作","时间"],
	["考勤"],
	["请假"],
	["租房","补贴"],
	["托儿所","幼儿园","入托","费用","补贴"],
	["高温"],
	["午餐","餐补","补贴"],
	["通讯","补贴"],
	["过节","福利"],
	["社保","查询"],
	["生育","保险","领取"],
	["公积金","查询"],
	["公积金","缴费","比例"],
	["公积金","提取","条件"],
	["兼职"]
]

len_answers=len(answers)

allwords = set(itertools.chain(*keywords))

len_allwords = len(allwords)

vtoi = {v:i for i,v in enumerate(allwords)}
itov = {i:v for v,i in vtoi.items()}


def get_tf():
	_tf = np.zeros((len_allwords,len_answers),dtype = np.float64)
	for i,d in enumerate(keywords):
		counter = Counter(d)
		for v in counter.keys():
			_tf[vtoi[v],i] = counter[v] / counter.most_common(1)[0][1]

	return np.log(1+_tf)

def get_idf():
	df = np.zeros((len_allwords,1))
	for i in range(len_allwords):
		dcount = 0
		for d in keywords:
			dcount += 1 if itov[i] in d else 0
		df [i,0] = dcount

	return 1+np.log(len_answers/(df+1))

tf = get_tf()
idf = get_idf()

tf_idf = tf * idf

def cosine_similarity(q_vector,_tf_idf):
	vecq = q_vector/np.sqrt(np.sum(np.square(q_vector),axis=0, keepdims=True))					#[n_vcab,1]
	vecall = _tf_idf/np.sqrt(np.sum(np.square(_tf_idf),axis=0,keepdims=True))					#[n_vcab,n_docs]
	product = vecall.T.dot(vecq).ravel()

	return product

def answerSequence(q):
	unkown_v = 0																			#the number of new vcab
	v2i = vtoi.copy()
	i2v = itov.copy()
	for v in set(q):
		if v not in v2i:
			v2i[v] = len(v2i)
			i2v[len(v2i)-1] = v
			unkown_v += 1
	if unkown_v:
		_idf = np.concatenate((idf,np.zeros((unkown_v,1),dtype = np.float64)),axis =0)
		_tf_idf = np.concatenate((tf_idf,np.zeros((unkown_v,tf_idf.shape[1]),dtype = np.float64)),axis = 0)
	else:
		_idf,_tf_idf = idf,tf_idf

	counter = Counter(q)
	q_tf = np.zeros((len(_idf),1),dtype = np.float64)
	for v in counter.keys():
		q_tf[v2i[v],0] = counter[v]

	q_vector = q_tf * _idf																	#[n_vcab,1]
	answerSequence = cosine_similarity(q_vector,_tf_idf)
	return answerSequence



while  True:
	q = input("输入问题:  ")
	if q == "quit":
		break
	else:
		q = jieba.lcut(q,cut_all=True)
		answer = answerSequence(q)
		top = answer.argsort()[-1:]
		print(answers[top[0]])