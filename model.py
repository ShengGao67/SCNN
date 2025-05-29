#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import numpy as np
import math
import mesa
from mesa import Agent, Model
from mesa.time import BaseScheduler
import pandas as pd
from SCN.agent import Order, Customer, Manufacturer, Supplier, manhattan_distance


# In[5]:


def update_transport_cost_for_order(order, model):
    if order.order_type == "product" and order.completed:
        # 计算运输成本：距离 * 订单数量
        transport_cost = order.distance *0.2 * order.quantity * (1+0.5* (order.logistics_speed/4-1))
        # 计算违约成本：若实际送达天数大于理论送达天数，则额外惩罚 = 延误天数 * 订单数量 * 1
        penalty_cost = 0
        if order.actual_delivery_time > order.theoretical_delivery_time:
            delay = order.actual_delivery_time - order.theoretical_delivery_time
            penalty_cost = delay * order.quantity * 5
        # 查找对应的 Manufacturer（订单的 receiver_id）
        for m in model.manufacturers:
            if m.unique_id == order.receiver_id:
                m.cumulative_transport_cost += transport_cost
                m.cumulative_penalty_cost += penalty_cost
                m.completed_products += order.quantity
                break

GRID_SIZE = 20

'''used_positions = set()

def random_position_unique(pos_list):
    """
    从 pos_list 中随机取一个位置 (x, y)，并将其移除，避免重复使用。
    """
    pos = random.choice(pos_list)
    pos_list.remove(pos)
    return pos
'''

# In[6]:


class SupplyChainGridModel(Model):
    def __init__(self, 
                 cust_positions,
                 man_positions,
                 sup_positions,
                 demands_list,          # 客户需求量列表
                 inventory_list,        # 初始库存列表
                 num_customers=10,
                 num_manufacturers=6,
                 num_suppliers=2,
                 cust_demand_probability=0.3,
                 cust_demand_multiplier=1.0, 
                 m_production_capacity=10,
                 m_inventory_capacity_product=400,
                 m_inventory_capacity_material=800,
                 m_status='Y',
                 s_material_capacity=40,
                 agent_mode="homogeneous",  # 可选值："homogeneous", "heterogeneous", "regional_heterogeneous"
                 logistics_speed=1.0,
                 # Manufacturer 原材料采购策略参数：
                 rm_procurement_mode="gap_based",  # "gap_based" 或 "reorder_point"
                 rm_reorder_point=30,
                 rm_reorder_target=500,
                 rm_purchase_multiplier=1.8,
                 rm_produce_multiplier=1.7,
                # 新增参数：产品订单模式和原材料订单模式
                 product_order_mode="normal",         # 用于 Customer："normal" 或 "multi_m"
                 material_order_mode="normal"           # 用于 Manufacturer："normal" 或 "multi_s"
                ): 
        super().__init__()
        self.schedule = BaseScheduler(self)
        self.pending_deliveries = []
        self.delivered_orders = []
        self.steps = 0
        self.logistics_speed = logistics_speed
        self.agent_mode = agent_mode
        self.distances_initialized = False
        self.grid_width = 21

        # =========== 创建 Customer ===========
        self.customers = []
        for i in range(num_customers):
            pos = cust_positions[i]
            customer = Customer(
                unique_id=f"C{i+1}",
                model=self,
                pos=pos,
                demands_list=demands_list,
                demand_probability=cust_demand_probability,
                product_order_mode=product_order_mode,
                cust_demand_multiplier=cust_demand_multiplier
            )
            self.customers.append(customer)
            self.schedule.add(customer)

        # =========== 创建 Manufacturer ===========
        self.manufacturers = []
        for i in range(num_manufacturers):
            pos = man_positions[i]
            # 根据 m_cap_mode 设置 Manufacturer 的 production_capacity
            if agent_mode == "homogeneous":
                prod_cap = m_production_capacity
            elif agent_mode == "heterogeneous":
                prod_cap = random.randint(max(1, m_production_capacity - 30), m_production_capacity + 30)
            elif agent_mode == "regional_heterogeneous":
                # 若所在位置在网格右侧五列，则产能异质；否则同质
                if pos[0] >= self.grid_width - 5:
                    prod_cap = random.randint(max(1, m_production_capacity - 30), m_production_capacity + 30)
                else:
                    prod_cap = m_production_capacity
            else:
                raise ValueError("Invalid m_cap_mode value.")

            init_prod_inv = random.choice(inventory_list)
            init_raw_inv = random.choice(inventory_list)

            manufacturer = Manufacturer(
                unique_id=f"M{i+1}",
                model=self,
                pos=pos,
                status=m_status,
                production_capacity=prod_cap,
                initial_product_inventory=init_prod_inv,
                initial_raw_material_inventory=init_raw_inv,
                inventory_capacity_product=m_inventory_capacity_product,
                inventory_capacity_material=m_inventory_capacity_material,
                rm_procurement_mode=rm_procurement_mode,
                rm_reorder_point=rm_reorder_point,
                rm_reorder_target=rm_reorder_target,
                material_order_mode=material_order_mode,
                rm_purchase_multiplier=rm_purchase_multiplier
            )
            self.manufacturers.append(manufacturer)
            self.schedule.add(manufacturer)

        # =========== 创建 Supplier ===========
        self.suppliers = []
        for i in range(num_suppliers):
            pos = sup_positions[i]
            init_raw_material = random.choice(inventory_list) * 2
            supplier = Supplier(
                unique_id=f"S{i+1}",
                model=self,
                pos=pos,
                material_capacity=s_material_capacity,
                initial_raw_material_inventory=init_raw_material
            )
            self.suppliers.append(supplier)
            self.schedule.add(supplier)
    def _compute_all_distances(self):
        agents = list(self.schedule.agents)
        customers     = [a for a in agents if isinstance(a, Customer)]
        manufacturers = [a for a in agents if isinstance(a, Manufacturer)]
        suppliers     = [a for a in agents if isinstance(a, Supplier)]
        '''for customer in self.customers:
            print(f"Customer {customer.unique_id} at {customer.pos}")
        for m in self.manufacturers:
            print(f"Manufacturer {m.unique_id} at {m.pos}")
        for s in self.suppliers:
            print(f"Supplier {s.unique_id} at {s.pos}")'''
        # Customer ↔ Manufacturer
        for c in customers:
            for m in manufacturers:
                d = manhattan_distance(c.pos, m.pos)
                # 同时存到 Customer 和 Manufacturer
                c.distance_to_manufacturer[m.unique_id] = d
                m.distance_to_customer[c.unique_id]   = d
        # Manufacturer ↔ Supplier
        for m in manufacturers:
            for s in suppliers:
                d = manhattan_distance(m.pos, s.pos)
                m.distance_to_supplier[s.unique_id]    = d
                s.distance_to_manufacturer[m.unique_id] = d

    def step(self):
        if not self.distances_initialized:
            self._compute_all_distances()
            self.distances_initialized = True
            
        self.schedule.step()
        # 遍历 pending_deliveries 中所有产品订单
        for order in list(self.pending_deliveries):
            if order.order_type == "product":
                order.update(self.schedule.steps)
                if order.completed:
                    # 找到订单的接收方（例如 Customer）
                    receiver = next((agent for agent in self.schedule.agents if agent.unique_id == order.receiver_id), None)
                    if receiver and isinstance(receiver, Customer):
                        receiver.receive_delivery(order)
                    # 更新制造商统计信息（例如运输成本等）
                    update_transport_cost_for_order(order, self)
                    self.delivered_orders.append(order)
                    self.pending_deliveries.remove(order)
        self.steps += 1


# In[7]:


def calculate_unit_cost(model):
    """
    计算并返回当前模型下所有Manufacturer的平均单位成本：
      - total_cost = 所有Manufacturer的 (累计仓储 + 运输 + 违约) 成本之和
      - total_completed = 所有Manufacturer的完成产品数量之和
    返回值：如果 total_completed > 0，则返回 (total_cost / total_completed)；否则返回 None。
    """
    total_cost = 0
    total_completed = 0
    for m in model.manufacturers:
        m_total_cost = (
            m.cumulative_storage_cost_product +
            m.cumulative_storage_cost_material +
            m.cumulative_transport_cost +
            m.cumulative_penalty_cost
        )
        total_cost += m_total_cost
        total_completed += m.completed_products

    if total_completed > 0:
        return total_cost / total_completed
    else:
        return None


# In[ ]:




