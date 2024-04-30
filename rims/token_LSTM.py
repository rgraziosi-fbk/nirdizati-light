from datetime import timedelta
import simpy
import pm4py
import random

from rims.checking_process import SimulationProcess
from pm4py.objects.petri_net import semantics
from rims.MAINparameters import*


class Token(object):

    def __init__(self, id, params, process: SimulationProcess, case, sequence, contrafactual, rp_feature='all_role'):
        self.id = id
        self.net, self.am, self.fm = pm4py.read_pnml(params.PATH_PETRINET)
        self.process = process
        self.start_time = params.START_SIMULATION
        self.pr_wip_initial = params.PR_WIP_INITIAL
        self.rp_feature = rp_feature
        self.params = params
        self.see_activity = False
        self.case = case
        self.pos = 0
        self.prefix = []
        self.sequence = sequence
        self.contrafactual = contrafactual
        self.CF = True if self.contrafactual else False

    def read_json(self, path):
        with open(path) as file:
            data = json.load(file)
            self.ac_index = data['ac_index']

    def simulation(self, env: simpy.Environment, writer, type, syn=False):
        #trans = self.next_transition(syn)
        event = self.next_event() if self.contrafactual==False else self.next_event_contrafactual()
        ### register trace in process ###
        resource_trace = self.process.get_resource_trace()
        resource_trace_request = resource_trace.request()

        while event is not None:
            yield resource_trace_request

            buffer = [self.id, event[0]]
            buffer.append(str(self.start_time + timedelta(seconds=env.now)))
            ### call predictor for waiting time
            #print(event)
            role = self.params.RESOURCE_ROLE[str(event[2])]
            resource = self.process.get_single_resource(str(event[2]))  ## ruolo

            transition = (self.params.INDEX_AC[event[0]], self.params.INDEX_ROLE[role])
            self.prefix.append(event[0])
            pr_wip_wait = self.pr_wip_initial + resource_trace.count
            #rp_oc = self.process.get_occupations_resource(role)
            rp_oc = self.process.get_occupations_all_role(role)
            request_resource = resource.request_CF() if self.contrafactual else resource.request_original()
            if len(resource.queue) > 0:
                queue = len(resource.queue[-1])
            else:
                queue = 0
            waiting = self.process.get_predict_waiting(str(self.id), pr_wip_wait, transition, rp_oc,
                                                       self.start_time + timedelta(seconds=env.now), -1)
            if self.contrafactual is not None:
                if self.see_activity:
                    yield env.timeout(waiting)
            else:
                yield env.timeout(float(event[3]))
            yield request_resource

            ### register event in process ###
            resource_task = self.process.get_resource_event(event[0])
            resource_task_request = resource_task.request()
            yield resource_task_request
            buffer.append(str(self.start_time + timedelta(seconds=env.now)))
            ### call predictor for processing time
            pr_wip = self.pr_wip_initial + resource_trace.count
            #rp_oc = self.process.get_occupations_resource(resource.get_name())
            rp_oc = self.process.get_occupations_all_role(role)
            initial_ac_wip = self.params.AC_WIP_INITIAL[event[0]] if event[0] in self.params.AC_WIP_INITIAL else 0
            ac_wip = initial_ac_wip + resource_task.count

            duration = self.process.get_predict_processing(str(self.id), pr_wip, transition, ac_wip, rp_oc, self.start_time + timedelta(seconds=env.now), -1)
            if event[1] == -1:
                yield env.timeout(duration)
            else:
                if event[1] < 0:
                    event[1] = 0
                yield env.timeout(event[1])
            buffer.append(str(self.start_time + timedelta(seconds=env.now)))
            buffer.append(resource.get_name())
            buffer.append(pr_wip_wait)
            buffer.append(ac_wip)
            buffer.append(queue)
            #if self.contrafactual is not False and self.contrafactual!=[]:
            #    buffer.append(self.contrafactual[0][2])
            #else:
            #    print(self.id, event)
            buffer.append(event[4])
            resource.release(request_resource)
            resource_task.release(resource_task_request)
            print(*buffer)
            writer.writerow(buffer)

            #self.update_marking(trans)
            #trans = self.next_transition(syn)
            self.see_activity = True
            event = self.next_event() if self.contrafactual == False else self.next_event_contrafactual()

        resource_trace.release(resource_trace_request)


    def next_event_contrafactual(self):
        if self.contrafactual:
            if self.sequence and self.sequence[0][0] == self.contrafactual[0][0]:
                next = self.sequence[0]
            else:
                # [event, event_processingTime, resource, wait, amount]
                next = [self.contrafactual[0][0], -1, self.contrafactual[0][1], -1, self.contrafactual[0][2]]
            if self.sequence:
                del self.sequence[0]
            del self.contrafactual[0]
        else:
            next = None
        return next

    def next_event(self):
        if self.sequence:
            next = self.sequence[0]
            del self.sequence[0]
        else:
            next = None
        return next

    def update_marking(self, trans):
        self.am = semantics.execute(trans, self.net, self.am)

    def next_transition(self, syn):
        all_enabled_trans = semantics.enabled_transitions(self.net, self.am)
        all_enabled_trans = list(all_enabled_trans)
        all_enabled_trans.sort(key=lambda x: x.name)
        label_element = str(list(self.am)[0])
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) > 1:
            if syn:
                prob = [0.50, 0.50]
                if label_element in ['exi_node_54ded9af-1e77-4081-8659-bd5554ae9b9d', 'exi_node_38c10378-0c54-4b13-8c4c-db3e4d952451', 'exi_node_52e223db-6ebf-4cc7-920e-9737fe97b655', 'exi_node_e55f8171-c3fc-4120-9ab1-167a472007b7']:
                    prob = [0.01, 0.99]
                elif label_element in ['exi_node_7af111c0-b232-4481-8fce-8d8278b1cb5a']:
                    prob = [0.99, 0.01]
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
            else:
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
            return all_enabled_trans[next]
        else:
            return all_enabled_trans[0]
