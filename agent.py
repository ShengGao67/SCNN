#!/usr/bin/env python
# coding: utf-8

# In[17]:


import random
import numpy as np
import math
import mesa
from mesa import Agent
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# In[18]:


class Order:
    """
    Order 类：
      - order_id: 订单唯一编号
      - creation_time: 订单创建时间
      - sender_id, receiver_id: 发货/收货方 ID
      - material_id: 物料类型 ("product" 或 "raw_material")
      - quantity: 需求量
      - order_type: "product" 或 "raw_material"
      - distance: 发货方与收货方的距离
      - shipped: 是否已从发货方仓库出库
      - completed: 是否已最终送达收货方
      - theoretical_delivery_time: 理论送达时间 = distance / LOGISTICS_SPEED
      - remaining_time: 初始即设为 theoretical_delivery_time；只有在订单发货后才开始倒计时
      - actual_delivery_time: 订单完成时，实际耗时 = 当前时间步 - creation_time
    """
    def __init__(self, order_id, creation_time, sender_id, receiver_id,
                 material_id, quantity, order_type, distance, logistics_speed):
        self.order_id = order_id
        self.creation_time = creation_time
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.material_id = material_id
        self.quantity = quantity
        self.order_type = order_type
        self.distance = distance
        self.logistics_speed = logistics_speed

        self.shipped = False     # 是否已出库
        self.completed = False   # 是否已送达

        # 理论送达时间，单位：时间步
        self.theoretical_delivery_time = math.ceil(distance / 4)
        # 初始 remaining_time 就等于理论值
        self.remaining_time = math.ceil(distance / logistics_speed)

        # actual_delivery_time 从订单创建时开始计时
        self.actual_delivery_time = None

    def start_shipping(self, current_step):
        """
        当订单真正发货时调用：
          - 标记 shipped 为 True
          - 此时 remaining_time 开始倒计时
        """
        self.shipped = True
    def update(self, current_step):
        """
        每个时间步由模型调用：
          - 若订单已发货但未完成，则 remaining_time 递减 1；
          - 当 remaining_time <= 0 时，标记订单完成，
            并计算 actual_delivery_time = current_step - creation_time。
          - 若订单尚未发货，则 remaining_time 保持不变。
        """
        if self.shipped and not self.completed:
            self.remaining_time -= 1
            if self.remaining_time <= 0:
                self.remaining_time = 0
                self.completed = True
                self.actual_delivery_time = current_step - self.creation_time

    def __repr__(self):
        return (f"Order({self.order_id}, type={self.order_type},quantity={self.quantity},shipped={self.shipped}, "
                f"completed={self.completed}, remaining_time={self.remaining_time:.1f}, "
                f"theoretical={self.theoretical_delivery_time:.1f})")

# In[19]:


class Customer(Agent):
    """
    Customer 代理：
      - 每个时间步以一定概率产生一个产品订单（"product"），
      - 根据订单模式：
          * "normal" 模式：随机选择一个状态为 'Y' 的 Manufacturer 下单；
          * "multi_m" 模式：随机选择两个不同的状态为 'Y' 的 Manufacturer，
             每个订单的需求量为原需求的一半（如果订单需求量为奇数，则第一个订单多分配余数）。
      - 将订单加入模型的 pending_deliveries 供后续 update()。
    """
    def __init__(self, unique_id, model, pos, demands_list, demand_probability=0.4, product_order_mode="normal", cust_demand_multiplier=1.0):
        super().__init__(unique_id, model)
        self.pos = pos
        self.demands_list = demands_list
        self.demand_probability = demand_probability
        self.product_order_mode = product_order_mode  # "normal" 或 "multi_m"
        self.cust_demand_multiplier = cust_demand_multiplier
        self.order_counter = 0
        self.orders = {}  # 存储该 customer 发出的订单
        self.distance_to_manufacturer = {}

    def generate_order(self):
        """以一定概率产生订单，创建 Order 对象并发送给 Manufacturer"""
        if random.random() < self.demand_probability:
            # 1. 生成需求量
            demand = int(self.cust_demand_multiplier * random.choice(self.demands_list))
            self.order_counter += 1

            # 2. 全部 Manufacturer 和状态为 Y 的 Manufacturer
            all_ms = [a for a in self.model.schedule.agents if isinstance(a, Manufacturer)]
            if not all_ms:
                return
            ys = [m for m in all_ms if m.status == 'Y']

            # 用已有的 distance_to_manufacturer
            def pick_nearest(cands, k=1):
                lst = []
                for m in cands:
                    d = self.distance_to_manufacturer[m.unique_id]
                    lst.append((d, m))
                random.shuffle(lst)
                lst.sort(key=lambda x: x[0])
                return [m for _, m in lst[:k]]

            # --- Normal 模式 ---
            if self.product_order_mode == "normal":
                if ys:
                    chosen = random.choice(ys)
                else:
                    chosen = pick_nearest(all_ms, 1)[0]

                order_id = f"{self.unique_id}_{self.order_counter}_{chosen.unique_id}"
                distance = self.distance_to_manufacturer[chosen.unique_id]
                order = Order(
                    order_id=order_id,
                    creation_time=self.model.schedule.steps,
                    sender_id=self.unique_id,
                    receiver_id=chosen.unique_id,
                    material_id="product",
                    quantity=demand,
                    order_type="product",
                    distance=distance,
                    logistics_speed=self.model.logistics_speed
                )
                self.orders[order_id] = order
                chosen.receive_order(order)
                self.model.pending_deliveries.append(order)

            # --- multi_m 模式 ---
            else:
                # 先挑状态为Y的
                if len(ys) >= 2:
                    targets = random.sample(ys, 2)
                elif len(ys) == 1:
                    first = ys[0]
                    rest = [m for m in all_ms if m is not first]
                    second = pick_nearest(rest, 1)[0]
                    targets = [first, second]
                else:
                    targets = pick_nearest(all_ms, 2)

                # 将 demand 平均分配
                qty_each = demand // 2
                rem = demand - qty_each * 2  # 如果是奇数，第一个多分余数

                for idx, m in enumerate(targets, start=1):
                    qty = qty_each + (rem if idx == 1 else 0)
                    order_id = f"{self.unique_id}_{self.order_counter}_{idx+1}_{m.unique_id}"
                    distance = self.distance_to_manufacturer[m.unique_id]
                    order = Order(
                        order_id=order_id,
                        creation_time=self.model.schedule.steps,
                        sender_id=self.unique_id,
                        receiver_id=m.unique_id,
                        material_id="product",
                        quantity=qty,
                        order_type="product",
                        distance=distance,
                        logistics_speed=self.model.logistics_speed
                    )
                    self.orders[order_id] = order
                    m.receive_order(order)
                    self.model.pending_deliveries.append(order)
                
            '''manufacturers = [
                agent for agent in self.model.schedule.agents
                if isinstance(agent, Manufacturer)
            ]
            if not manufacturers:
                return 
            # 筛选状态为 'Y' 的 Manufacturer

            if self.product_order_mode == "normal":
                # 普通模式：随机选择一个 Manufacturer
                chosen_manufacturer = random.choice(manufacturers)
                order_id = f"{self.unique_id}_{self.order_counter}_{chosen_manufacturer.unique_id}"
                distance = self.distance_to_manufacturer[chosen_manufacturer.unique_id]
                #distance = manhattan_distance(self.pos, chosen_manufacturer.pos)
                
                order = Order(
                    order_id=order_id,
                    creation_time=self.model.schedule.steps,
                    sender_id=self.unique_id,
                    receiver_id=chosen_manufacturer.unique_id,
                    material_id="product",
                    quantity=demand,
                    order_type="product",
                    distance=distance,
                    logistics_speed=self.model.logistics_speed
                )
                self.orders[order_id] = order
                chosen_manufacturer.receive_order(order)
                self.model.pending_deliveries.append(order)
            elif self.product_order_mode == "multi_m":
                # multi_m 模式：随机选择两个不同的 Manufacturer
                if len(valid_manufacturers) < 2:
                    # 若可选制造商不足2个，退回到 normal 模式
                    chosen_manufacturer = random.choice(manufacturers)
                    order_id = f"{self.unique_id}_{self.order_counter}_{chosen_manufacturer.unique_id}"
                    distance = self.distance_to_manufacturer[chosen_manufacturer.unique_id]
                    order = Order(
                        order_id=order_id,
                        creation_time=self.model.schedule.steps,
                        sender_id=self.unique_id,
                        receiver_id=chosen_manufacturer.unique_id,
                        material_id="product",
                        quantity=demand,
                        order_type="product",
                        distance=distance,
                        logistics_speed=self.model.logistics_speed
                    )
                    self.orders[order_id] = order
                    chosen_manufacturer.receive_order(order)
                    self.model.pending_deliveries.append(order)
                else:
                    manufacturers = random.sample(valid_manufacturers, 2)
                    # 将 demand 平均分成两份
                    qty_each = demand // 2
                    remainder = demand - qty_each * 2  # 若 demand 为奇数，余数归于第一个订单
                    for idx, m in enumerate(manufacturers):
                        order_id = f"{self.unique_id}_{self.order_counter}_{idx+1}_{m.unique_id}"
                        distance = distance = self.distance_to_manufacturer[m.unique_id]
                        # 第一个订单数量加上余数
                        qty = qty_each + (remainder if idx == 0 else 0)
                        order = Order(
                            order_id=order_id,
                            creation_time=self.model.schedule.steps,
                            sender_id=self.unique_id,
                            receiver_id=m.unique_id,
                            material_id="product",
                            quantity=qty,
                            order_type="product",
                            distance=distance,
                            logistics_speed=self.model.logistics_speed
                        )
                        self.orders[order_id] = order
                        m.receive_order(order)
                        self.model.pending_deliveries.append(order)
                        '''
    def step(self):
        self.generate_order()
        
    def receive_delivery(self, order):
        """当订单最终送达时，可在此处记录或打印日志"""
        print(f"Customer {self.unique_id} received {order.order_id} at step {self.model.schedule.steps}, "
              f"actual_delivery_time={order.actual_delivery_time}, "
              f"theoretical={order.theoretical_delivery_time:.1f}")


# In[20]:


class Manufacturer(Agent):
    """
      1. 先来先发 (FIFO) 的发货逻辑，不进行部分交付。
      2. 排产逻辑：若 gap < 0，需要生产。1 产品需 2 单位原材料。- 若原材料不足，则消耗全部原材料进行部分生产，并向供应商下订单。
      3. 连续 5 个时间步 gap < 0，则 status = 'N'；否则 'Y'。
      4. 库存在发货和生产时实时更新：product_inventory, raw_material_inventory。
      5. 加入库存上限 (inventory_capacity)：每个时间步成品库存不能超过该上限，
         生产时也只能生产到库存达到上限为止。
    """
    def __init__(self, unique_id, model, pos, status,
                 production_capacity=100,
                 initial_product_inventory=50,
                 initial_raw_material_inventory=50,
                 inventory_capacity_product=400,
                 inventory_capacity_material=800,
                 # 新增原材料采购相关参数：
                 rm_procurement_mode="gap_based",  # "gap_based" 或 "reorder_point"
                 rm_reorder_point=30,
                 rm_reorder_target=500,
                 rm_purchase_multiplier=1.8,
                 rm_produce_multiplier=1.7,
                 material_order_mode="normal"      # 新增参数: "normal" 或 "multi_s"
                ):
        super().__init__(unique_id, model)
        self.pos = pos
        self.production_capacity = production_capacity
        self.status = 'Y'
        self.product_inventory = initial_product_inventory
        self.raw_material_inventory = initial_raw_material_inventory
        # 拆分库存上限：成品库存和原材料库存分别有各自上限
        self.inventory_capacity_product = inventory_capacity_product
        self.inventory_capacity_material = inventory_capacity_material


        self.pending_orders = {}
        self.current_produced = 0
        self.current_shipped = 0

        # 成本属性
        self.cumulative_storage_cost_product = 0
        self.cumulative_storage_cost_material = 0
        self.cumulative_transport_cost = 0
        self.cumulative_penalty_cost = 0
        self.completed_products = 0

        # 原材料采购策略参数
        self.rm_procurement_mode = rm_procurement_mode
        self.rm_reorder_point = rm_reorder_point
        self.rm_reorder_target = rm_reorder_target
        self.rm_purchase_multiplier = rm_purchase_multiplier
        self.rm_produce_multiplier = rm_produce_multiplier

        # 新增供应原材料订单下单模式参数
        self.material_order_mode = material_order_mode
        self.distance_to_customer = {}
        self.distance_to_supplier = {}

    def receive_order(self, order):
        if order.order_type == "product":
            self.pending_orders[order.order_id] = order

    def ship_orders(self):
        # 先来先发
        sorted_orders = sorted(self.pending_orders.values(), key=lambda o: o.creation_time)
        shipped_ids = []
        for order in sorted_orders:
            if not order.shipped:
                if self.product_inventory >= order.quantity:
                    # 扣减库存
                    self.product_inventory -= order.quantity
                    self.current_shipped += order.quantity
                    # 调用订单的 start_shipping，而非直接 order.shipped=True
                    order.start_shipping(self.model.schedule.steps)
                    shipped_ids.append(order.order_id)
                else:
                    break
        
        for s_id in shipped_ids:
            del self.pending_orders[s_id]

    def calculate_gap(self):
        backlog_demand = sum(o.quantity for o in self.pending_orders.values() if not o.shipped)
        gap = self.product_inventory - backlog_demand
        return gap
            
    def update_incoming_raw_material_orders(self):
        """
        检查全局 pending_deliveries 中所有属于本制造商下的原材料订单，
        若订单完成（completed==True），则将该订单的数量加到 raw_material_inventory 中，
        并从 pending_deliveries 中移除，记录到 delivered_orders 中。
        """
        for order in list(self.model.pending_deliveries):
            # 判断该订单是否为原材料订单，且是由本制造商下的订单
            if order.order_type == "raw_material" and order.sender_id == self.unique_id:
                # 更新订单状态（倒计时）
                order.update(self.model.schedule.steps)
                if order.completed:
                    self.raw_material_inventory += order.quantity
                    self.model.delivered_orders.append(order)
                    self.model.pending_deliveries.remove(order)
                    
    def production_logic(self):
        """
        排产逻辑：
          1. 计算 gap = product_inventory - backlog_demand，若 gap < 0，则需要生产。
          2. 考虑库存上限，生产数量不能使成品库存超过 inventory_capacity。
          3. 1 产品需 2 单位原材料：若原材料充足，则一次性生产；否则部分生产并下采购单补充原材料。
          4. 根据原材料采购模式进行补货下单：
             - 如果rm_procurement_mode=="gap_based"：当净原材料（库存+在途）不足 rm_purchase_multiplier * required_rm_for_gap 时下单。
             - 如果rm_procurement_mode=="reorder_point"：当原材料库存低于rm_reorder_point时，下单补齐至rm_reorder_target。
        """
        gap = self.calculate_gap()
        # 原材料生产时先检查采购需求
        # 计算在途原材料订单数量
        in_transit_rm = sum(o.quantity for o in self.model.pending_deliveries 
                            if o.order_type == "raw_material" and o.sender_id == self.unique_id)
        net_rm = self.raw_material_inventory + in_transit_rm

        if self.rm_procurement_mode == "gap_based":
            required_rm_for_gap = 2 * abs(gap)  # 缺口对应的原材料需求
            if net_rm < self.rm_purchase_multiplier * required_rm_for_gap:
                shortage = self.rm_purchase_multiplier * required_rm_for_gap - net_rm
                self.order_raw_material(shortage)
        elif self.rm_procurement_mode == "reorder_point":
            # 当库存低于补货点时，下单补足至目标库存水平
            if self.raw_material_inventory < self.rm_reorder_point:
                needed = self.rm_reorder_target - self.raw_material_inventory
                self.order_raw_material(needed)
        else:
            raise ValueError("Invalid rm_procurement_mode value.")

        # 生产逻辑
        if gap < 0:
            self.status = 'N'
            # 理论上需要生产的产品数量（也受生产能力限制）
            production_needed = min(self.rm_produce_multiplier * abs(gap), self.production_capacity)
            # 考虑库存上限
            available_space = self.inventory_capacity_product - self.product_inventory
            production_needed = min(production_needed, available_space)
            required_rm = 2 * production_needed
            if self.raw_material_inventory >= required_rm:
                self.product_inventory += production_needed
                self.raw_material_inventory -= required_rm
                self.current_produced = production_needed
            else:
                partial_production = self.raw_material_inventory // 2
                self.product_inventory += partial_production
                self.raw_material_inventory -= partial_production * 2
                self.current_produced = partial_production
        else:
            self.status = 'Y'
            self.current_produced = 0
    def order_raw_material(self, raw_material_needed):
        """
        下原材料订单：
          - 先检查原材料库存上限（inventory_capacity_material），只下差额部分。
          - 根据 material_order_mode 分为两种：
              * "normal" 模式：从单一 Supplier 下单。
              * "multi_s" 模式：随机选择两个不同的 Supplier，将订单拆分为两个部分订单。
        """
        capacity_diff = self.inventory_capacity_material - self.raw_material_inventory
        if capacity_diff <= 0:
            return  # 已达到库存上限
        # 实际下单数量不能超过可用容量
        raw_material_needed = min(raw_material_needed, capacity_diff)
        suppliers = [
            agent for agent in self.model.schedule.agents
            if isinstance(agent, Supplier) 
        ]
        if not suppliers:
            return
        
        # 选择可用的 Supplier
        '''valid_suppliers = [
            agent for agent in self.model.schedule.agents
            if isinstance(agent, Supplier) and agent.status == 'Y'
        ]
        # 如果没有可用供应商，就直接返回
        #if not valid_suppliers:
            return
        '''
        
        if self.material_order_mode == "normal": #or len(valid_suppliers) < 2:
            # "normal" 模式或可选供应商不足2个时，采用单一供应商下单
            chosen_supplier = random.choice(suppliers)
            order_id = f"{self.unique_id}_RM_{self.model.schedule.steps}"
            dist = self.distance_to_supplier[chosen_supplier.unique_id]
            raw_order = Order(
                order_id=order_id,
                creation_time=self.model.schedule.steps,
                sender_id=self.unique_id,
                receiver_id=chosen_supplier.unique_id,
                material_id="raw_material",
                quantity=raw_material_needed,
                order_type="raw_material",
                distance=dist,
                logistics_speed=self.model.logistics_speed
            )
            chosen_supplier.receive_order(raw_order)
            self.model.pending_deliveries.append(raw_order)
        elif self.material_order_mode == "multi_s":
            # multi_s 模式：随机选取两个不同的供应商，并将订单拆分为两个部分订单
            suppliers = random.sample(suppliers, 2)
            qty_each = raw_material_needed // 2
            remainder = raw_material_needed - qty_each * 2  # 若 raw_material_needed 不整除，将余数归给第一个订单
            for idx, s in enumerate(suppliers):
                order_id = f"{self.unique_id}_RM_{self.model.schedule.steps}_{idx+1}_{s.unique_id}"
                dist = self.distance_to_supplier[s.unique_id]
                # 第一个订单的数量加上余数
                qty = qty_each + (remainder if idx == 0 else 0)
                raw_order = Order(
                    order_id=order_id,
                    creation_time=self.model.schedule.steps,
                    sender_id=self.unique_id,
                    receiver_id=s.unique_id,
                    material_id="raw_material",
                    quantity=qty,
                    order_type="raw_material",
                    distance=dist,
                    logistics_speed=self.model.logistics_speed
                )
                s.receive_order(raw_order)
                self.model.pending_deliveries.append(raw_order)
        else:
            raise ValueError("Invalid material_order_mode value.")

    def step(self):
        self.update_incoming_raw_material_orders()
        self.ship_orders()
        gap = self.calculate_gap()
        #self.update_status(gap)
        self.production_logic()
        self.current_shipped = 0 
        # 累计当前时刻的仓储成本（单位仓储成本为1）
        self.cumulative_storage_cost_product += self.product_inventory*1
        self.cumulative_storage_cost_material += self.raw_material_inventory*0.4


# In[21]:


class Supplier(Agent):
    """
    Supplier 代理，逻辑与 Manufacturer 一致：
      1. 存储未发货订单 pending_orders，先来先发 (FIFO) 不部分交付。
      2. 使用 raw_material_inventory 表示可生产或已产出的原材料库存。
      3. 若 gap < 0 则生产 (不需向上游下单)，生产量 = min(abs(gap), production_capacity)。
      4. 连续 5 个时间步 gap < 0，则 status='N'，否则='Y'。
    """
    def __init__(self, unique_id, model, pos,
                 material_capacity=100,
                 initial_raw_material_inventory=50):
        super().__init__(unique_id, model)
        self.pos = pos
        self.material_capacity = material_capacity
        #self.status = 'Y'
        
        self.raw_material_inventory = initial_raw_material_inventory
        self.pending_orders = {}
        self.gap_history = []
        
        self.current_produced = 0
        self.current_shipped = 0
        self.distance_to_manufacturer = {}

    def receive_order(self, order):
        if order.order_type == "raw_material":
            self.pending_orders[order.order_id] = order

    def ship_orders(self):
        sorted_orders = sorted(self.pending_orders.values(), key=lambda o: o.creation_time)
        shipped_ids = []
        for order in sorted_orders:
            if not order.shipped:
                if self.raw_material_inventory >= order.quantity:
                    self.raw_material_inventory -= order.quantity
                    self.current_shipped += order.quantity
                    # 开始发货
                    order.start_shipping(self.model.schedule.steps)
                    shipped_ids.append(order.order_id)
                else:
                    break
        for s_id in shipped_ids:
            del self.pending_orders[s_id]

    def calculate_gap(self):
        backlog_demand = sum(o.quantity for o in self.pending_orders.values() if not o.shipped)
        return self.raw_material_inventory - backlog_demand

    '''def update_status(self, gap):
        self.gap_history.append(gap)
        if len(self.gap_history) > 10:
            self.gap_history.pop(0)
        if len(self.gap_history) == 10 and all(g < 0 for g in self.gap_history):
            self.status = 'N'
        else:
            self.status = 'Y'
    '''

    def production_logic(self):
        gap = self.calculate_gap()
        if gap < 0:
            production_needed = min(abs(gap), self.material_capacity)
            self.raw_material_inventory += production_needed
            self.current_produced = production_needed
        else:
            self.current_produced = 0

    def step(self):
        self.ship_orders()
        gap = self.calculate_gap()
        #self.update_status(gap)
        self.production_logic()
        self.current_shipped = 0


# In[ ]:




