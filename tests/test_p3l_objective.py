"""Independent vector algebra and dense complete-candidate CUDA oracles for P3l."""
import copy
import math
import unittest
try:
    import torch
except ImportError:
    torch = None
from experiments.p3l_objective import gradient_metrics, joint_backward
from experiments.p3j_objective import centered_backward
import test_p3j_objective as historical


@unittest.skipUnless(torch is not None, 'torch required')
class VectorTests(unittest.TestCase):
    def test_orthogonal_aligned_opposed_and_all_parameters(self):
        for direction in (1., -1., 0.):
            d = [torch.tensor([3., 0.], requires_grad=True), torch.tensor([4.])]
            c = [torch.tensor([direction*6., 8.]), torch.tensor([direction*8.])]
            m = gradient_metrics(d, c)
            expected_d = 5.
            expected_c = math.sqrt(64.+100.*direction**2)
            self.assertAlmostEqual(m['d'], expected_d)
            self.assertAlmostEqual(m['c'], expected_c)
            self.assertAlmostEqual(m['alpha'], .5/expected_c)
            self.assertIs(type(m['alpha']), float)
            self.assertIsNone(d[0].grad)
            self.assertAlmostEqual(m['auxiliary_to_dmc_ratio'], .1)
            self.assertGreaterEqual(m['joint_descent_factor'], .9)

    def test_exact_parallel_and_antiparallel_bound(self):
        d=[torch.tensor([3.]),torch.tensor([4.])]
        for sign,factor in ((1.,1.1),(-1.,.9)):
            m=gradient_metrics(d,[sign*2.*g for g in d])
            self.assertAlmostEqual(m['cosine'],sign)
            self.assertAlmostEqual(m['joint_descent_factor'],factor)
            self.assertAlmostEqual(m['alpha'],.05)

    def test_cap_zero_threshold_and_constant(self):
        def metrics(d,c,arm='normcap'):
            return gradient_metrics([torch.tensor([d],dtype=torch.float64)],
                                    [torch.tensor([c],dtype=torch.float64)],arm)
        m=metrics(2.,.01);self.assertEqual(m['alpha'],1.);self.assertTrue(m['cap_triggered'])
        self.assertEqual(m['alpha_branch'],'cap')
        for d,c,branch in ((0.,3.,'zero_dmc'),(1e-12,3.,'zero_dmc'),(3.,1e-12,'zero_centered')):
            m=metrics(d,c);self.assertEqual(m['alpha'],0.);self.assertEqual(m['alpha_branch'],branch)
        self.assertIsNone(metrics(0.,3.)['auxiliary_to_dmc_ratio'])
        self.assertEqual(metrics(3.,0.)['auxiliary_to_dmc_ratio'],0.)
        self.assertIsNone(metrics(3.,0.)['cosine'])
        self.assertEqual(metrics(2.,.01,'constant')['alpha'],.1)

    def test_nonfinite_and_malformed_rejected(self):
        for v in (math.nan,math.inf,-math.inf):
            with self.assertRaises(ValueError):gradient_metrics([torch.tensor([v])],[torch.tensor([1.])])
            with self.assertRaises(ValueError):gradient_metrics([torch.tensor([1.])],[torch.tensor([v])])
        with self.assertRaises(ValueError):gradient_metrics([],[])
        with self.assertRaises(ValueError):gradient_metrics([torch.ones(2)],[torch.ones(1)])
        with self.assertRaises(ValueError):gradient_metrics([torch.ones(2)],[torch.ones(2)],'bad')


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
class CUDATests(unittest.TestCase):
    setUp=historical.CUDATests.setUp
    requests=historical.CUDATests.requests
    dense=historical.CUDATests.dense

    def oracle(self,model,requests):
        aux,dmc=self.dense(model,requests)
        parameters=list(model.parameters())
        gd=torch.autograd.grad(dmc,parameters,retain_graph=True)
        gc=torch.autograd.grad(aux,parameters,retain_graph=True)
        # Independent dense float64 concatenation, not the implementation reducer.
        dv=torch.cat([g.detach().flatten().double() for g in gd])
        cv=torch.cat([g.detach().flatten().double() for g in gc])
        d,c=dv.norm().item(),cv.norm().item()
        alpha=0. if d<=1e-12 or c<=1e-12 else min(1.,.1*d/c)
        (dmc+alpha*aux).backward()
        return aux.item(),d,c,alpha

    def test_dense_joint_chunks_singleton_denominator_and_one_adam(self):
        req=self.requests()
        self.assertEqual(len(req[-1][1]),1)
        for chunk in (1,7,1024):
            a,b=copy.deepcopy(self.model),copy.deepcopy(self.model)
            _,dmc=self.dense(a,req);dmc.backward()
            result=joint_backward(a,req,'normcap',chunk)
            loss,d,c,alpha=self.oracle(b,req)
            self.assertEqual(result['requests'],4)
            self.assertEqual(result['state_denominator'],4)
            self.assertEqual(result['candidate_sizes'],[len(x[1]) for x in req])
            self.assertEqual(result['single_candidate_requests'],1)
            self.assertEqual(result['candidates'],sum(len(x[1]) for x in req))
            self.assertAlmostEqual(result['loss'],loss,places=6)
            for k,v in (('d',d),('c',c),('alpha',alpha)):
                self.assertAlmostEqual(result[k],v,delta=max(1e-8,abs(v)*2e-5))
            for x,y in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)
            oa=torch.optim.Adam(a.parameters(),lr=.001,capturable=True)
            ob=torch.optim.Adam(b.parameters(),lr=.001,capturable=True)
            oa.step();ob.step()
            for x,y in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
            self.assertTrue(all(s['step'].item()==1 for s in oa.state.values()))

    def test_constant_exact_gradient_and_adam(self):
        req=self.requests()
        for chunk in (1,7,1024):
            a,b=copy.deepcopy(self.model),copy.deepcopy(self.model)
            for m in (a,b):
                _,dmc=self.dense(m,req);dmc.backward()
            joint_backward(a,req,'constant',chunk)
            centered_backward(b,req,.1,chunk)
            for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x.grad,y.grad))
            oa=torch.optim.Adam(a.parameters(),lr=.001,capturable=True)
            ob=torch.optim.Adam(b.parameters(),lr=.001,capturable=True)
            oa.step();ob.step()
            for x,y in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(x,y))
                for key in oa.state[x]:self.assertTrue(torch.equal(oa.state[x][key],ob.state[y][key]))

    def test_singleton_zero_branch_and_missing_nonfinite_gradients(self):
        req=self.requests()[-1:]
        _,dmc=self.dense(self.model,req);dmc.backward()
        before=[p.grad.clone() for p in self.model.parameters()]
        result=joint_backward(self.model,req,'normcap',7)
        self.assertEqual(result['alpha'],0.)
        self.assertEqual(result['alpha_branch'],'zero_centered')
        for p,g in zip(self.model.parameters(),before):self.assertTrue(torch.equal(p.grad,g))
        next(self.model.parameters()).grad.flatten()[0]=math.nan
        with self.assertRaises(ValueError):joint_backward(self.model,req)
        self.model.zero_grad(set_to_none=True)
        with self.assertRaises(ValueError):joint_backward(self.model,req)
        for p in self.model.parameters():p.grad=torch.zeros_like(p)
        result=joint_backward(self.model,self.requests(),'normcap',7)
        self.assertEqual(result['alpha_branch'],'zero_dmc')
        self.assertIsNone(result['auxiliary_to_dmc_ratio'])
        self.assertTrue(all(p.grad.count_nonzero().item()==0 for p in self.model.parameters()))

    def test_8769_full_candidates_dense_joint(self):
        from benchmark_learning_runtime import high_branch_fixture
        req=[high_branch_fixture()]
        a,b=self.model,copy.deepcopy(self.model)
        _,dmc=self.dense(a,req);dmc.backward()
        result=joint_backward(a,req,'normcap',1024)
        self.oracle(b,req)
        self.assertEqual(result['candidates'],8769)
        self.assertEqual(result['candidate_sizes'],[8769])
        for x,y in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)

    def test_nonfinite_centered_failure_does_not_update_parameters(self):
        from unittest.mock import patch
        req=self.requests();_,dmc=self.dense(self.model,req);dmc.backward()
        before=copy.deepcopy(self.model.state_dict())
        with patch.object(self.model,'forward',side_effect=lambda states,actions: torch.full((len(actions),),math.nan,device='cuda')):
            with self.assertRaises(ValueError):joint_backward(self.model,req)
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))


